import pickle
import socket
import struct
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor
# import gymnasium as gym

from .socket_env_base import SocketEnv

class SocketEnvClient(SocketEnv):

    def __init__(self, socket_path=None):
        super().__init__(socket_path)

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self._socket_path)

    @property
    def observation_space(self):
        if self.obs_buffer is None:
            raise RuntimeError("Environment not initialized.")
        return self.obs_buffer
        # return gym.spaces.Box(
        #     low=-float("inf"), high=float("inf"), shape=self.obs_buffer.shape, dtype=self.obs_buffer.dtype
        # )
    
    @property
    def action_space(self):
        if self.action_buffer is None:
            raise RuntimeError("Environment not initialized.")
        return self.action_buffer
        # return gym.spaces.Box(
        #     low=-1.0, high=1.0, shape=self.action_buffer.shape, dtype=self.action_buffer.dtype
        # )
    
    @property
    def num_envs(self):
        if self.rewards_buffer is None:
            raise RuntimeError("Environment not initialized.")
        return self.rewards_buffer.shape[0]

    def _socket_send_receive(self, sig_send: bytes):
        self.sock.sendall(sig_send)
        # Wait for server response 
        sig_rec = self.sock.recv(1)
        if sig_rec != sig_send:
            raise RuntimeError(f"Expected signal {sig_send}, got {sig_rec}")

    def _receive_tensor_metadata(self) -> torch.Tensor:
        """Receive IPC metadata from server and reconstruct the shared CUDA tensor."""
        (length,) = struct.unpack("!I", self._recv_exact(self.sock, 4))
        meta = pickle.loads(self._recv_exact(self.sock, length))
        tensor = rebuild_cuda_tensor(**meta)
        return tensor
    
    def _receive_buffers_metadata(self):
        self.obs_buffer = self._receive_tensor_metadata()
        self.rewards_buffer = self._receive_tensor_metadata()
        self.terminated_buffer = self._receive_tensor_metadata()
        self.truncated_buffer = self._receive_tensor_metadata()
        self.action_buffer = self._receive_tensor_metadata()

    def make(self, task: str, num_envs: int):
        self.sock.sendall(self.MAKE)
        # Send task configuration (could be extended to send more complex configs)
        task_bytes = task.encode("utf-8")
        payload = struct.pack("!I", len(task_bytes)) + task_bytes + struct.pack("!i", num_envs)
        self.sock.sendall(payload)

        self._receive_buffers_metadata()

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

    def stop(self):
        self._socket_send_receive(self.STOP)
        self.sock.close()