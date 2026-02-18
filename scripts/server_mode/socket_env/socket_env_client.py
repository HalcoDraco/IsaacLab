import socket
import struct
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor
# import gymnasium as gym

from .socket_env_base import SocketEnv

class SocketEnvClient(SocketEnv):

    def __init__(self, socket_path=None):
        super().__init__(socket_path)

        self._observation_space = None
        self._action_space = None

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

    def make(self, task: str, num_envs: int):
        if self.sock is None:
            raise RuntimeError("Socket is not connected.")
        self.sock.sendall(self.MAKE)
        # Send task configuration (could be extended to send more complex configs)
        task_bytes = task.encode("utf-8")
        payload = (
            struct.pack("!I", len(task_bytes))
            + task_bytes
            + struct.pack("!i", num_envs)
        )
        self.sock.sendall(payload)

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
        self._socket_send_receive(self.CLOSE)
        self._observation_space = None
        self._action_space = None

    def stop(self):
        self._socket_send_receive(self.STOP)
        if self.sock is None:
            raise RuntimeError("Socket is not connected.")
        self.sock.close()