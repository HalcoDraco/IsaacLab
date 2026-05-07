import socket
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor
import gymnasium as gym
import os

from .socket_env_base import SocketEnv

class SocketEnvClient(SocketEnv):

    def __init__(self, socket_path=None):
        super().__init__(socket_path)

        self._isaac_task: str | None = None
        self._num_envs: int | None = None
        self._observation_space: gym.spaces.Space | None = None
        self._action_space: gym.spaces.Space | None = None
        self._state_space: gym.spaces.Space | None = None
        self._asymmetric_obs: bool | None = None
        self._env_cfg_overrides: dict | None = None
        self._generate_video: bool | None = None
        self._max_episode_length: int | None = None
        self.is_made = False

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self._socket_path)

    @property
    def isaac_task(self):
        if not self.is_made:
            raise RuntimeError("Environment not initialized.")
        return self._isaac_task

    @property
    def observation_space(self):
        if not self.is_made:
            raise RuntimeError("Environment not initialized.")
        return self._observation_space
    
    @property
    def action_space(self):
        if not self.is_made:
            raise RuntimeError("Environment not initialized.")
        return self._action_space

    @property
    def state_space(self):
        if not self.is_made:
            raise RuntimeError("Environment not initialized.")
        return self._state_space

    @property
    def asymmetric_obs(self):
        if not self.is_made:
            raise RuntimeError("Environment not initialized.")
        return self._asymmetric_obs
    
    @property
    def num_envs(self):
        if not self.is_made:
            raise RuntimeError("Environment not initialized.")
        return self._num_envs
    
    @property
    def env_cfg_overrides(self):
        if not self.is_made:
            raise RuntimeError("Environment not initialized.")
        return self._env_cfg_overrides

    @property
    def generate_video(self):
        if not self.is_made:
            raise RuntimeError("Environment not initialized.")
        return self._generate_video

    @property
    def max_episode_length(self):
        if not self.is_made:
            raise RuntimeError("Environment not initialized.")
        return self._max_episode_length

    def _socket_send_receive(self, sig_send: bytes):
        if self._sock is None:
            raise RuntimeError("Socket is not connected.")
        self._sock.sendall(sig_send)
        # Wait for server response 
        sig_rec = self._sock.recv(1)
        if sig_rec != sig_send:
            raise RuntimeError(f"Expected signal {sig_send}, got {sig_rec}")

    def _receive_tensor_metadata(self) -> torch.Tensor:
        """Receive IPC metadata from server and reconstruct the shared CUDA tensor."""
        meta = self._receive_pickled_object()
        tensor = rebuild_cuda_tensor(**meta)
        return tensor
    
    def _receive_buffers_metadata(self):
        self._obs_buffer = self._receive_tensor_metadata()
        self._state_buffer = self._receive_tensor_metadata()
        self._rewards_buffer = self._receive_tensor_metadata()
        self._terminated_buffer = self._receive_tensor_metadata()
        self._truncated_buffer = self._receive_tensor_metadata()
        self._action_buffer = self._receive_tensor_metadata()

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
        if self._sock is None:
            raise RuntimeError("Socket is not connected.")
        self._sock.sendall(self.MAKE)
        self._send_pickled_object(
            {
                "task": task,
                "num_envs": num_envs,
                "env_cfg_overrides": env_cfg_overrides or {},
                "generate_video": generate_video,
            }
        )

        self._receive_buffers_metadata()
        self._isaac_task = task
        self._num_envs = num_envs
        self._env_cfg_overrides = env_cfg_overrides
        self._generate_video = generate_video
        self.is_made = True
        self._observation_space = self._receive_pickled_object()
        self._action_space = self._receive_pickled_object()
        self._max_episode_length = self._receive_pickled_object()
        self._asymmetric_obs = self._receive_pickled_object()
        self._state_space = self._receive_pickled_object()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Take an environment step on the server with the given actions.
        
        Parameters
        ----------
        actions: torch.Tensor
            A tensor of shape (num_envs, action_dim) containing the actions to take in each environment.

        Returns
        -------
        obs: torch.Tensor
            A tensor of shape (num_envs, obs_dim) containing the observations after taking the actions.
        rewards: torch.Tensor
            A tensor of shape (num_envs,) containing the rewards received after taking the actions.
        terminated: torch.Tensor
            A boolean tensor of shape (num_envs,) indicating which environments have terminated.
        truncated: torch.Tensor
            A boolean tensor of shape (num_envs,) indicating which environments have been truncated due to reaching the maximum episode length.
        extras: dict
            A dict containing extra data. If asymmetric observations are enabled, this includes
            ``{"full_state": <tensor>}``.
        """

        if self._obs_buffer is None or \
            self._rewards_buffer is None or \
            self._terminated_buffer is None or \
            self._truncated_buffer is None or \
            self._action_buffer is None:
            raise RuntimeError("Buffers not initialized.")
        
        self._action_buffer.copy_(actions)
        torch.cuda.synchronize()
        self._socket_send_receive(self.STEP)

        extras: dict = {}
        if self._asymmetric_obs:
            extras = {"full_state": self._state_buffer}

        return self._obs_buffer, self._rewards_buffer, self._terminated_buffer, self._truncated_buffer, extras

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset the environments on the server.

        Returns
        -------
        obs: torch.Tensor
            A tensor of shape (num_envs, obs_dim) containing the initial observations after reset.
        extras: dict
            A dict containing extra data. If asymmetric observations are enabled, this includes
            ``{"full_state": <tensor>}``.
        """
        
        if self._obs_buffer is None:
            raise RuntimeError("Observation buffer not initialized.")
        
        self._socket_send_receive(self.RESET)

        extras: dict = {}
        if self._asymmetric_obs:
            extras = {"full_state": self._state_buffer}
        return self._obs_buffer, extras

    def close(self):
        """Close the current environment on the server.

        The socket stays open so you can call :meth:`make` again to create a
        new environment.  To fully disconnect, call :meth:`disconnect`.
        """
        self._socket_send_receive(self.CLOSE)

        # Release shared-memory buffer references so they can be freed.
        self._obs_buffer = None
        self._state_buffer = None
        self._rewards_buffer = None
        self._terminated_buffer = None
        self._truncated_buffer = None
        self._action_buffer = None

        self._isaac_task = None
        self._num_envs = None
        self._env_cfg_overrides = None
        self._generate_video = None
        self._max_episode_length = None
        self._observation_space = None
        self._action_space = None
        self._state_space = None
        self._asymmetric_obs = None
        self.is_made = False
        
    def disconnect(self):
        """Close the environment (if open) and shut down the socket.

        After this call the server sees a clean EOF and goes back to
        waiting for a new client.  To interact again, create a new
        :class:`SocketEnvClient`.
        """
        if self._sock is None:
            return

        # Cleanly shut down the socket so the server sees EOF immediately
        # instead of a connection-reset error.
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()
        self._sock = None

    def stop(self):
        self._socket_send_receive(self.STOP)
        if self._sock is None:
            raise RuntimeError("Socket is not connected.")
        self._sock.close()

    def save_video(self, path: str):
        """Copy the video generated by the server to the given relative path.
        
        Args:
            path (str): The relative path with respect to the root of the repository,
            where the video should be saved.
        """
        BASE_REPO_DIR = "/workspace"
        RELATIVE_SERVER_VIDEO_PATH = "isaaclab_fork/IsaacLab/scripts/server_mode/videos/rl-video-step-0.mp4"
        server_video_path = os.path.join(BASE_REPO_DIR, RELATIVE_SERVER_VIDEO_PATH)
        target_path = os.path.join(BASE_REPO_DIR, path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        os.replace(server_video_path, target_path)
        