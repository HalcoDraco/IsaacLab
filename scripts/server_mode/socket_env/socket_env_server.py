import os
import pickle
import socket
import struct
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from .socket_env_base import SocketEnv

class SocketEnvServer(SocketEnv):

    def __init__(self, socket_path=None):
        super().__init__(socket_path)
        self.env: gym.Env | None = None

        self.srv: socket.socket | None = None
        self.conn: socket.socket | None = None

    def _prepare_socket(self):
        os.makedirs(os.path.dirname(self._socket_path), exist_ok=True)
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)

        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(self._socket_path)
        self.srv.listen(1)

    def _wait_for_client(self):
        if self.srv is None:
            raise RuntimeError("Socket server not initialized.")
        
        self.conn, _ = self.srv.accept()
        

    def _send_tensor_metadata(self, tensor: torch.Tensor):
        if self.conn is None:
            raise RuntimeError("No client connected.")
        meta = pickle.dumps(self._get_tensor_metadata(tensor))
        self.conn.sendall(struct.pack("!I", len(meta)) + meta)

    def _send_buffers_metadata(self):
        if self.obs_buffer is None or \
            self.rewards_buffer is None or \
            self.terminated_buffer is None or \
            self.truncated_buffer is None or \
            self.action_buffer is None:
            raise RuntimeError("Buffers not initialized.")
        self._send_tensor_metadata(self.obs_buffer)
        self._send_tensor_metadata(self.rewards_buffer)
        self._send_tensor_metadata(self.terminated_buffer)
        self._send_tensor_metadata(self.truncated_buffer)
        self._send_tensor_metadata(self.action_buffer)

    def _make(self):
        # Receive task configuration from client
        if self.conn is None:
            raise RuntimeError("No client connected.")

        (task_len,) = struct.unpack("!I", self._recv_exact(self.conn, 4))
        task = self._recv_exact(self.conn, task_len).decode("utf-8")
        (num_envs,) = struct.unpack("!i", self._recv_exact(self.conn, 4))

        env_cfg = parse_env_cfg(
            task, device=self.device, 
            num_envs=num_envs, 
            use_fabric=not self.disable_fabric
        )
        # create environment
        self.env = gym.make(task, cfg=env_cfg)

        # Create buffers for IPC sharing
        self.obs_buffer = torch.empty(self.env.observation_space.shape, device=self.device)
        self.rewards_buffer = torch.empty((num_envs,), device=self.device)
        self.terminated_buffer = torch.empty((num_envs,), dtype=torch.bool, device=self.device)
        self.truncated_buffer = torch.empty((num_envs,), dtype=torch.bool, device=self.device)
        self.action_buffer = torch.empty(self.env.action_space.shape, device=self.device)

        torch.cuda.synchronize()
        self._send_buffers_metadata()

    def _step(self):
        if self.env is None:
            raise RuntimeError("Environment not initialized.")
        if self.conn is None:
            raise RuntimeError("No client connected.")
        if self.obs_buffer is None or \
            self.rewards_buffer is None or \
            self.terminated_buffer is None or \
            self.truncated_buffer is None or \
            self.action_buffer is None:
            raise RuntimeError("Buffers not initialized.")
        
        obs_dict, rewards, terminated, truncated, infos = self.env.step(self.action_buffer)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        self.obs_buffer.copy_(obs)
        self.rewards_buffer.copy_(rewards)
        self.terminated_buffer.copy_(terminated)
        self.truncated_buffer.copy_(truncated)
        torch.cuda.synchronize()
        self.conn.sendall(self.STEP)

    def _reset(self):
        if self.env is None:
            raise RuntimeError("Environment not initialized.")
        if self.conn is None:
            raise RuntimeError("No client connected.")
        if self.obs_buffer is None:
            raise RuntimeError("Observation buffer not initialized.")
        
        obs_dict, _ = self.env.reset()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        self.obs_buffer.copy_(obs)
        torch.cuda.synchronize()
        self.conn.sendall(self.RESET)

    def _close(self):
        if self.env is None:
            raise RuntimeError("Environment not initialized.")
        if self.conn is None:
            raise RuntimeError("No client connected.")
        
        self.env.close()
        torch.cuda.synchronize()

        self.env = None
        self.obs_buffer = None
        self.rewards_buffer = None
        self.terminated_buffer = None
        self.truncated_buffer = None
        self.action_buffer = None

        self.conn.sendall(self.CLOSE)

    def run(self):
        try:
            self._prepare_socket()
            print("[isaac_server] Waiting for connection…")
            self._wait_for_client()
            assert self.conn is not None
            print("[isaac_server] Client connected.")

            while True:
                sig = self.conn.recv(1)
                if sig == self.MAKE:
                    self._make()
                elif sig == self.STEP:
                    self._step()
                elif sig == self.RESET:
                    self._reset()
                elif sig == self.CLOSE:
                    self._close()
                    print("[isaac_server] Environment closed by client.")
                    break
                else:
                    raise ValueError(f"Unknown signal received: {sig}")
        finally:
            torch.cuda.synchronize()
            if self.obs_buffer is not None:
                del self.obs_buffer
            if self.rewards_buffer is not None:
                del self.rewards_buffer
            if self.terminated_buffer is not None:
                del self.terminated_buffer
            if self.truncated_buffer is not None:
                del self.truncated_buffer
            if self.action_buffer is not None:
                del self.action_buffer
            if self.conn is not None:
                self.conn.close()
            if self.srv is not None:
                self.srv.close()
            print("[isaac_server] Done.")