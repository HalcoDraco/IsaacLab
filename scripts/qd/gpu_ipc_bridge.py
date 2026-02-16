"""GPU IPC Bridge: Zero-copy CUDA tensor sharing between a Docker container and host process.

This module provides a protocol for sharing GPU tensors between an Isaac Lab Docker container
and a host process using CUDA IPC. It enables a training algorithm running on the host to
interact with an Isaac Lab environment running inside Docker, with zero-copy GPU memory access
for observations and actions.

Architecture:
    - Container side (IsaacEnvServer): Runs the Isaac Lab env, allocates shared CUDA buffers,
      and serves env steps over TCP + GPU IPC.
    - Host side (IsaacEnvClient): Connects to the server, opens shared CUDA buffers,
      and provides a gym-like interface for RL/QD algorithms.

Protocol:
    1. Server allocates CUDA buffers for obs, actions, rewards, dones, truncated.
    2. Server sends IPC handles to client over TCP (MSG_INIT).
    3. Client opens the same GPU memory via CUDA IPC (zero copy).
    4. Step loop:
       a. Server writes obs to GPU buffer -> sends MSG_OBS_READY
       b. Client reads obs, computes action, writes to action buffer -> sends MSG_ACT_READY
       c. Server reads action, calls env.step(), writes results -> sends MSG_OBS_READY
       d. Repeat

Requirements:
    - PyTorch with CUDA support on both sides
    - Docker container started with ipc: host (use docker-compose.gpu-ipc.yaml overlay)
    - Same GPU accessible from both container and host (default with nvidia runtime)
    - No additional dependencies beyond PyTorch and Python standard library

Usage:
    Container (inside Docker):
        python scripts/qd/isaac_env_server.py --task <TASK> --num_envs <N> --headless

    Host:
        from gpu_ipc_bridge import IsaacEnvClient
        client = IsaacEnvClient(port=5555)
        obs = client.reset()
        obs, reward, done, truncated, info = client.step(action)
        client.close()
"""

from __future__ import annotations

import logging
import pickle
import socket
import struct

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

# Message types (1 byte each)
MSG_INIT = b"\x01"  # Server -> Client: buffer metadata + IPC handles
MSG_OBS_READY = b"\x02"  # Server -> Client: observation is ready in GPU buffer
MSG_ACT_READY = b"\x03"  # Client -> Server: action is ready in GPU buffer
MSG_CLOSE = b"\x04"  # Either -> Either: request graceful shutdown
MSG_RESET = b"\x05"  # Client -> Server: request full env reset

DEFAULT_PORT = 5555
DEFAULT_HOST = "127.0.0.1"

# Header: 1 byte msg_type + 4 bytes payload length (network byte order)
_HEADER_FMT = "!cI"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 5 bytes


# ---------------------------------------------------------------------------
# TCP messaging helpers
# ---------------------------------------------------------------------------


def send_msg(sock: socket.socket, msg_type: bytes, payload: dict | None = None) -> None:
    """Send a typed, length-prefixed message over TCP.

    Args:
        sock: Connected TCP socket.
        msg_type: One of the MSG_* constants.
        payload: Optional dict to pickle and send as the message body.
    """
    body = pickle.dumps(payload) if payload is not None else b""
    header = struct.pack(_HEADER_FMT, msg_type, len(body))
    sock.sendall(header + body)


def recv_msg(sock: socket.socket) -> tuple[bytes, dict | None]:
    """Receive a typed, length-prefixed message from TCP.

    Returns:
        (msg_type, payload) tuple. payload is None if body is empty.

    Raises:
        ConnectionError: If the connection is closed unexpectedly.
    """
    header = _recv_exact(sock, _HEADER_SIZE)
    if header is None:
        raise ConnectionError("Connection closed while reading header")
    msg_type, length = struct.unpack(_HEADER_FMT, header)
    if length > 0:
        body = _recv_exact(sock, length)
        if body is None:
            raise ConnectionError("Connection closed while reading payload")
        payload = pickle.loads(body)
    else:
        payload = None
    return msg_type, payload


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Receive exactly *n* bytes from a socket, or None on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------
# CUDA IPC tensor sharing
# ---------------------------------------------------------------------------


def get_tensor_ipc_info(tensor: torch.Tensor) -> dict:
    """Extract CUDA IPC metadata from a tensor so it can be opened in another process.

    Uses PyTorch's internal ``_share_cuda_()`` which wraps ``cudaIpcGetMemHandle``.
    The returned dict is pickle-safe and can be sent over a socket.

    Args:
        tensor: A **contiguous** CUDA tensor.

    Returns:
        A dict containing everything needed by :func:`open_ipc_tensor` to reconstruct
        the tensor in another process with zero-copy shared GPU memory.
    """
    if not tensor.is_cuda:
        raise ValueError("Tensor must be on a CUDA device")
    if not tensor.is_contiguous():
        raise ValueError("Tensor must be contiguous")

    storage = tensor.untyped_storage()
    # Returns: (device, handle, storage_size_bytes, storage_offset_bytes,
    #           ref_counter_handle, ref_counter_offset, event_handle, event_sync_required)
    ipc_data = storage._share_cuda_()  # type: ignore[attr-defined]  # intentional private API

    return {
        "ipc": ipc_data,
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def open_ipc_tensor(info: dict) -> torch.Tensor:
    """Open a shared CUDA tensor from IPC metadata produced by :func:`get_tensor_ipc_info`.

    Uses PyTorch's internal ``_new_shared_cuda()`` which wraps ``cudaIpcOpenMemHandle``.

    Args:
        info: Dict returned by :func:`get_tensor_ipc_info`.

    Returns:
        A CUDA tensor backed by the **same physical GPU memory** as the original tensor
        in the other process (zero copy).
    """
    dtype_str = info["dtype"].replace("torch.", "")
    dtype = getattr(torch, dtype_str)
    shape = tuple(info["shape"])
    device_idx = info["ipc"][0]

    # Reconstruct shared storage from the IPC handle
    storage = torch.UntypedStorage._new_shared_cuda(*info["ipc"])  # type: ignore[attr-defined]  # intentional private API

    # Compute contiguous strides
    strides: list[int] = []
    s = 1
    for dim in reversed(shape):
        strides.append(s)
        s *= dim
    strides.reverse()

    # Build tensor on the shared storage (same pattern as torch.multiprocessing.reductions)
    tensor = torch.empty(0, dtype=dtype, device=f"cuda:{device_idx}")
    tensor.set_(storage, 0, shape, tuple(strides))
    return tensor


# ---------------------------------------------------------------------------
# Server (runs inside the Docker container)
# ---------------------------------------------------------------------------


class IsaacEnvServer:
    """TCP server that exposes an Isaac Lab environment over GPU IPC.

    The server allocates CUDA buffers, extracts their IPC handles, and enters a
    request-response loop: it writes observations/rewards/dones to the shared
    GPU buffers and signals the client over TCP; the client writes actions and
    signals back.
    """

    def __init__(self, env, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """
        Args:
            env: A fully-initialised Isaac Lab gym environment.
            host: TCP bind address. Use ``"0.0.0.0"`` to accept connections from
                  any interface (needed when the client is on the host and the
                  server is inside a ``network_mode: host`` container).
            port: TCP port to listen on.
        """
        self.env = env
        self.host = host
        self.port = port
        self.device = env.unwrapped.device

        # --- Discover env dimensions ---
        unwrapped = env.unwrapped
        self.num_envs: int = unwrapped.num_envs

        obs_space = env.observation_space
        if hasattr(obs_space, "spaces"):
            # gymnasium.spaces.Dict — take the "policy" group
            policy_space = obs_space["policy"] if "policy" in obs_space.spaces else list(obs_space.spaces.values())[0]
            self.obs_size: int = policy_space.shape[-1]
        else:
            self.obs_size = obs_space.shape[-1]

        self.act_size: int = env.action_space.shape[-1]
        logger.info(
            f"Env dimensions: num_envs={self.num_envs}, obs_size={self.obs_size}, act_size={self.act_size}, "
            f"device={self.device}"
        )

        # --- Allocate shared CUDA buffers ---
        self.obs_buf = torch.zeros(self.num_envs, self.obs_size, dtype=torch.float32, device=self.device)
        self.act_buf = torch.zeros(self.num_envs, self.act_size, dtype=torch.float32, device=self.device)
        self.reward_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.done_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.truncated_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # --- Pre-compute IPC handles (pickle-safe) ---
        self._init_payload: dict = {
            "num_envs": self.num_envs,
            "obs_size": self.obs_size,
            "act_size": self.act_size,
            "buffers": {
                "obs": get_tensor_ipc_info(self.obs_buf),
                "act": get_tensor_ipc_info(self.act_buf),
                "reward": get_tensor_ipc_info(self.reward_buf),
                "done": get_tensor_ipc_info(self.done_buf),
                "truncated": get_tensor_ipc_info(self.truncated_buf),
            },
        }

    # ----- helpers -----

    def _extract_obs(self, obs) -> torch.Tensor:
        """Pull the flat policy-observation tensor from the env output."""
        if isinstance(obs, dict):
            return obs.get("policy", next(iter(obs.values())))
        return obs

    # ----- main loop -----

    def serve(self) -> None:
        """Start serving the environment.  Blocks until the client disconnects or sends MSG_CLOSE."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        logger.info(f"Server listening on {self.host}:{self.port} — waiting for client …")

        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        logger.info(f"Client connected from {addr}")

        try:
            self._serve_loop(conn)
        except ConnectionError as exc:
            logger.info(f"Client disconnected ({exc})")
        finally:
            conn.close()
            srv.close()
            self.env.close()
            logger.info("Server shut down")

    def _serve_loop(self, conn: socket.socket) -> None:
        """Inner request-response loop."""
        # Initial reset
        obs, _info = self.env.reset()
        obs_tensor = self._extract_obs(obs)
        self.obs_buf.copy_(obs_tensor)
        torch.cuda.synchronize(self.device)

        # Send init metadata (IPC handles, env dimensions)
        send_msg(conn, MSG_INIT, self._init_payload)
        # Signal that the first observation is ready
        send_msg(conn, MSG_OBS_READY)

        step = 0
        while True:
            msg_type, _payload = recv_msg(conn)

            if msg_type == MSG_ACT_READY:
                # The client has written an action into self.act_buf.
                # Ensure the GPU write from the client side is visible.
                torch.cuda.synchronize(self.device)

                # Step the simulation
                obs, reward, terminated, truncated, _info = self.env.step(self.act_buf)

                # Populate shared buffers
                self.obs_buf.copy_(self._extract_obs(obs))
                self.reward_buf.copy_(reward)
                self.done_buf.copy_(terminated.float())
                self.truncated_buf.copy_(truncated.float())
                torch.cuda.synchronize(self.device)

                send_msg(conn, MSG_OBS_READY)
                step += 1

            elif msg_type == MSG_RESET:
                obs, _info = self.env.reset()
                self.obs_buf.copy_(self._extract_obs(obs))
                self.reward_buf.zero_()
                self.done_buf.zero_()
                self.truncated_buf.zero_()
                torch.cuda.synchronize(self.device)
                send_msg(conn, MSG_OBS_READY)
                step = 0

            elif msg_type == MSG_CLOSE:
                logger.info(f"Client requested close after {step} steps")
                break
            else:
                logger.warning(f"Unknown message type: {msg_type!r}")


# ---------------------------------------------------------------------------
# Client (runs on the host)
# ---------------------------------------------------------------------------


class IsaacEnvClient:
    """Client that connects to an Isaac Lab env server via GPU IPC.

    Provides a gym-like interface backed by zero-copy shared CUDA buffers.
    The GPU memory holding observations, rewards, etc. is physically the same
    memory written by the container — no copies cross the process boundary.

    The ``step()`` method returns **clones** of the shared buffers so that the
    caller can safely store them without worrying about the next step
    overwriting the data.  For maximum performance, access ``self.obs_buf``
    (etc.) directly, but be aware that they are overwritten on every step.

    Example::

        client = IsaacEnvClient(port=5555)
        obs = client.reset()
        for _ in range(1000):
            action = policy(obs)
            obs, reward, done, truncated, info = client.step(action)
        client.close()
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self._connected = False
        self._sock: socket.socket | None = None

        # Populated after connect()
        self.num_envs: int = 0
        self.obs_size: int = 0
        self.act_size: int = 0
        self.obs_buf: torch.Tensor | None = None
        self.act_buf: torch.Tensor | None = None
        self.reward_buf: torch.Tensor | None = None
        self.done_buf: torch.Tensor | None = None
        self.truncated_buf: torch.Tensor | None = None

    # ----- connection lifecycle -----

    def connect(self) -> None:
        """Connect to the server and set up shared GPU buffers."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        logger.info(f"Connecting to {self.host}:{self.port} …")
        self._sock.connect((self.host, self.port))

        # Receive initialisation message
        msg_type, payload = recv_msg(self._sock)
        if msg_type != MSG_INIT:
            raise RuntimeError(f"Expected MSG_INIT from server, got {msg_type!r}")
        assert payload is not None, "MSG_INIT must carry a payload"

        self.num_envs = payload["num_envs"]
        self.obs_size = payload["obs_size"]
        self.act_size = payload["act_size"]

        # Open shared CUDA buffers (zero-copy IPC)
        bufs = payload["buffers"]
        self.obs_buf = open_ipc_tensor(bufs["obs"])
        self.act_buf = open_ipc_tensor(bufs["act"])
        self.reward_buf = open_ipc_tensor(bufs["reward"])
        self.done_buf = open_ipc_tensor(bufs["done"])
        self.truncated_buf = open_ipc_tensor(bufs["truncated"])

        self._connected = True
        logger.info(
            f"Connected — num_envs={self.num_envs}, obs={self.obs_buf.shape}, "
            f"act={self.act_buf.shape}, device={self.obs_buf.device}"
        )

    def close(self) -> None:
        """Close the connection to the server."""
        if self._connected and self._sock is not None:
            try:
                send_msg(self._sock, MSG_CLOSE)
            except Exception:
                pass
            self._sock.close()
            self._connected = False
            logger.info("Client disconnected")

    # ----- gym-like interface -----

    def reset(self) -> torch.Tensor:
        """Connect (if needed) and wait for the first observation.

        Returns:
            obs: ``(num_envs, obs_size)`` float32 CUDA tensor (clone of shared buffer).
        """
        if not self._connected:
            self.connect()
        assert self._sock is not None

        # Wait for the initial OBS_READY that the server sends after env.reset()
        msg_type, _ = recv_msg(self._sock)
        if msg_type != MSG_OBS_READY:
            raise RuntimeError(f"Expected OBS_READY after connect, got {msg_type!r}")

        torch.cuda.synchronize()
        assert self.obs_buf is not None
        return self.obs_buf.clone()

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Send an action and receive the next transition.

        Args:
            action: ``(num_envs, act_size)`` tensor.  Will be copied to the shared
                    action buffer.  Must be on a CUDA device (or CPU — it will be
                    moved automatically).

        Returns:
            obs:       ``(num_envs, obs_size)`` float32 — cloned from shared buffer.
            reward:    ``(num_envs,)`` float32.
            done:      ``(num_envs,)`` float32 (1.0 = terminated).
            truncated: ``(num_envs,)`` float32 (1.0 = truncated).
            info:      Empty dict (reserved for future use).
        """
        assert self._sock is not None
        assert self.act_buf is not None

        # Write action to the shared GPU buffer
        if not action.is_cuda:
            action = action.to(device=self.act_buf.device)
        self.act_buf.copy_(action)
        torch.cuda.synchronize()

        # Signal the server
        send_msg(self._sock, MSG_ACT_READY)

        # Wait for the next observation
        msg_type, _ = recv_msg(self._sock)
        if msg_type == MSG_CLOSE:
            raise ConnectionError("Server closed the connection")
        if msg_type != MSG_OBS_READY:
            raise RuntimeError(f"Expected OBS_READY, got {msg_type!r}")

        torch.cuda.synchronize()
        assert self.obs_buf is not None
        assert self.reward_buf is not None
        assert self.done_buf is not None
        assert self.truncated_buf is not None
        return (
            self.obs_buf.clone(),
            self.reward_buf.clone(),
            self.done_buf.clone(),
            self.truncated_buf.clone(),
            {},
        )

    def request_reset(self) -> torch.Tensor:
        """Request an explicit full environment reset.

        Note:
            Isaac Lab environments auto-reset terminated/truncated sub-environments
            inside ``step()``, so this is only needed if you want a full reset of
            *all* environments at once.

        Returns:
            obs: ``(num_envs, obs_size)`` float32 CUDA tensor.
        """
        assert self._sock is not None
        send_msg(self._sock, MSG_RESET)

        msg_type, _ = recv_msg(self._sock)
        if msg_type != MSG_OBS_READY:
            raise RuntimeError(f"Expected OBS_READY after reset, got {msg_type!r}")

        torch.cuda.synchronize()
        assert self.obs_buf is not None
        return self.obs_buf.clone()
