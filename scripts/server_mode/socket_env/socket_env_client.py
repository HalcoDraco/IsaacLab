import socket
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor
import gymnasium as gym

from .socket_env_base import SocketEnv

class SocketEnvClient(SocketEnv):

    def __init__(self, socket_path=None):
        super().__init__(socket_path)

        self._observation_space: gym.spaces.Space = None
        self._action_space: gym.spaces.Space = None

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self._socket_path)

    @property
    def observation_space(self):
        if self._observation_space is None:
            raise RuntimeError("Environment not initialized.")
        return self._observation_space
    
    @property
    def action_space(self):
        if self._action_space is None:
            raise RuntimeError("Environment not initialized.")
        return self._action_space
    
    @property
    def num_envs(self):
        if self.rewards_buffer is None:
            raise RuntimeError("Environment not initialized.")
        return self.rewards_buffer.shape[0]

    def _socket_send_receive(self, sig_send: bytes):
        if self.sock is None:
            raise RuntimeError("Socket is not connected.")
        self.sock.sendall(sig_send)
        # Wait for server response 
        sig_rec = self.sock.recv(1)
        if sig_rec != sig_send:
            raise RuntimeError(f"Expected signal {sig_send}, got {sig_rec}")

    def _receive_tensor_metadata(self) -> torch.Tensor:
        """Receive IPC metadata from server and reconstruct the shared CUDA tensor."""
        meta = self._receive_pickled_object()
        tensor = rebuild_cuda_tensor(**meta)
        return tensor
    
    def _receive_buffers_metadata(self):
        self.obs_buffer = self._receive_tensor_metadata()
        self.rewards_buffer = self._receive_tensor_metadata()
        self.terminated_buffer = self._receive_tensor_metadata()
        self.truncated_buffer = self._receive_tensor_metadata()
        self.action_buffer = self._receive_tensor_metadata()

    def make(
        self,
        task: str,
        num_envs: int,
        env_cfg_overrides: dict | None = None,
        generate_video: bool = False,
    ):
        """Create an environment on the server.

        Args:
            task: The registered gym task id (e.g. ``"Isaac-Cartpole-Direct-v0"``).
            num_envs: Number of parallel environments.
            env_cfg_overrides: Optional dict of config overrides applied on top
                of the task's default ``EnvCfg``.  Keys may use dot-separated
                paths for nested attributes, e.g.
                ``{"episode_length_s": 10.0, "sim.dt": 1/240}``.
            generate_video: If True, the server will generate videos of the environment.
        """
        if self.sock is None:
            raise RuntimeError("Socket is not connected.")
        self.sock.sendall(self.MAKE)
        self._send_pickled_object(
            {
                "task": task,
                "num_envs": num_envs,
                "env_cfg_overrides": env_cfg_overrides or {},
                "generate_video": generate_video,
            }
        )

        self._receive_buffers_metadata()
        self._observation_space = self._receive_pickled_object()
        self._action_space = self._receive_pickled_object()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.obs_buffer is None or \
            self.rewards_buffer is None or \
            self.terminated_buffer is None or \
            self.truncated_buffer is None or \
            self.action_buffer is None:
            raise RuntimeError("Buffers not initialized.")
        
        self.action_buffer.copy_(actions)
        torch.cuda.synchronize()
        self._socket_send_receive(self.STEP)

        return self.obs_buffer, self.rewards_buffer, self.terminated_buffer, self.truncated_buffer

    def reset(self) -> torch.Tensor:
        
        if self.obs_buffer is None:
            raise RuntimeError("Observation buffer not initialized.")
        
        self._socket_send_receive(self.RESET)
        return self.obs_buffer

    def close(self):
        """Close the current environment on the server.

        The socket stays open so you can call :meth:`make` again to create a
        new environment.  To fully disconnect, call :meth:`disconnect`.
        """
        self._socket_send_receive(self.CLOSE)
        self._observation_space = None
        self._action_space = None

        # Release shared-memory buffer references so they can be freed.
        self.obs_buffer = None
        self.rewards_buffer = None
        self.terminated_buffer = None
        self.truncated_buffer = None
        self.action_buffer = None

    def disconnect(self):
        """Close the environment (if open) and shut down the socket.

        After this call the server sees a clean EOF and goes back to
        waiting for a new client.  To interact again, create a new
        :class:`SocketEnvClient`.
        """
        if self.sock is None:
            return

        # Cleanly shut down the socket so the server sees EOF immediately
        # instead of a connection-reset error.
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        self.sock = None

    def stop(self):
        self._socket_send_receive(self.STOP)
        if self.sock is None:
            raise RuntimeError("Socket is not connected.")
        self.sock.close()