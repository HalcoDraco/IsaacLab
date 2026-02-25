import gc
import os
import socket
import torch
import gymnasium as gym
import traceback

import isaaclab.sim as sim_utils
import omni.physx
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab.envs import DirectRLEnv

from .socket_env_base import SocketEnv

class SocketEnvServer(SocketEnv):

    def __init__(self, socket_path=None, disable_fabric: bool = False):
        super().__init__(socket_path, disable_fabric=disable_fabric)
        self.env: DirectRLEnv | None = None

        self.srv: socket.socket | None = None

    def _prepare_socket(self):
        os.makedirs(os.path.dirname(self._socket_path), exist_ok=True)
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)

        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(self._socket_path)
        self.srv.listen(1)
        self.srv.settimeout(0.5)  # Set a timeout for accept() to allow graceful shutdown

    def _wait_for_client(self):
        if self.srv is None:
            raise RuntimeError("Socket server not initialized.")
        
        while True:
            try:
                self._sock, _ = self.srv.accept()
                break
            except socket.timeout:
                continue

    def _send_tensor_metadata(self, tensor: torch.Tensor):
        if self._sock is None:
            raise RuntimeError("No client connected.")
        self._send_pickled_object(self._get_tensor_metadata(tensor))

    def _send_buffers_metadata(self):
        if self._obs_buffer is None or \
            self._rewards_buffer is None or \
            self._terminated_buffer is None or \
            self._truncated_buffer is None or \
            self._action_buffer is None:
            raise RuntimeError("Buffers not initialized.")
        self._send_tensor_metadata(self._obs_buffer)
        self._send_tensor_metadata(self._rewards_buffer)
        self._send_tensor_metadata(self._terminated_buffer)
        self._send_tensor_metadata(self._truncated_buffer)
        self._send_tensor_metadata(self._action_buffer)

    def _make(self):
        # Receive task configuration from client
        if self._sock is None:
            raise RuntimeError("No client connected.")
        
        if self.env is not None:
            self._cleanup_env()

        make_request: dict = self._receive_pickled_object()
        task = make_request["task"]
        num_envs = make_request["num_envs"]
        env_cfg_overrides: dict = make_request.get("env_cfg_overrides", {})

        print(f"[isaac_server] Received make request: task={task}, num_envs={num_envs}, env_cfg_overrides={env_cfg_overrides}")

        env_cfg = parse_env_cfg(
            task, device=self.device, 
            num_envs=num_envs, 
            use_fabric=not self.disable_fabric
        )

        # Apply client-supplied config overrides (dot-separated keys supported)
        for key, value in env_cfg_overrides.items():
            parts = key.split(".")
            obj = env_cfg
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
            print(f"[isaac_server] Config override: {key} = {value}")

        # create environment
        generate_video = make_request.get("generate_video", False)
        self.env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if generate_video else None)

        # Optionally wrap with RecordVideo for server-side video capture
        if generate_video:
            video_kwargs = {
                "video_folder": f"{os.getcwd()}/scripts/server_mode/videos",
                "step_trigger": lambda step: step == 0,
                "video_length": 0,
                "disable_logger": True,
            }
            print(f"[isaac_server] Wrapping env with RecordVideo: {video_kwargs}")
            self.env = gym.wrappers.RecordVideo(self.env, **video_kwargs)

        # Create buffers for IPC sharing
        self._obs_buffer = torch.empty(self.env.observation_space.shape, device=self.device)
        self._rewards_buffer = torch.empty((num_envs,), device=self.device)
        self._terminated_buffer = torch.empty((num_envs,), dtype=torch.bool, device=self.device)
        self._truncated_buffer = torch.empty((num_envs,), dtype=torch.bool, device=self.device)
        self._action_buffer = torch.empty(self.env.action_space.shape, device=self.device)

        torch.cuda.synchronize()
        self._send_buffers_metadata()
        self._send_pickled_object(self.env.observation_space)
        self._send_pickled_object(self.env.action_space)
        self._send_pickled_object(self.env.unwrapped.max_episode_length)

    def _step(self):
        if self.env is None:
            raise RuntimeError("Environment not initialized.")
        if self._sock is None:
            raise RuntimeError("No client connected.")
        if self._obs_buffer is None or \
            self._rewards_buffer is None or \
            self._terminated_buffer is None or \
            self._truncated_buffer is None or \
            self._action_buffer is None:
            raise RuntimeError("Buffers not initialized.")
        
        obs_dict, rewards, terminated, truncated, infos = self.env.step(self._action_buffer)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        self._obs_buffer.copy_(obs)
        self._rewards_buffer.copy_(rewards)
        self._terminated_buffer.copy_(terminated)
        self._truncated_buffer.copy_(truncated)
        torch.cuda.synchronize()
        self._sock.sendall(self.STEP)

    def _reset(self):
        if self.env is None:
            raise RuntimeError("Environment not initialized.")
        if self._sock is None:
            raise RuntimeError("No client connected.")
        if self._obs_buffer is None:
            raise RuntimeError("Observation buffer not initialized.")
        
        obs_dict, _ = self.env.reset()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        self._obs_buffer.copy_(obs)
        torch.cuda.synchronize()
        self._sock.sendall(self.RESET)

    def _close(self):
        if self.env is None:
            raise RuntimeError("Environment not initialized.")
        if self._sock is None:
            raise RuntimeError("No client connected.")
        self._teardown_env(raise_on_error=True)

        self._sock.sendall(self.CLOSE)

    def _stop(self):
        if self._sock is None:
            raise RuntimeError("No client connected.")
        self._sock.sendall(self.STOP)

    def _teardown_env(self, raise_on_error: bool):
        """Tear down environment/simulator state.

        Args:
            raise_on_error: If True, propagate teardown exceptions.
                If False, swallow them and print warnings.
        """

        def _run_step(label: str, fn):
            if raise_on_error:
                fn()
            else:
                try:
                    fn()
                except Exception as e:
                    print(f"[isaac_server] Warning: Failed to {label}: {e}")

        # 1. Detach physics so env.close() / create_new_stage() don't hang.
        _run_step("detach stage", lambda: omni.physx.get_physx_simulation_interface().detach_stage())

        # 2. Close the Gym environment.
        if self.env is not None:
            _run_step("close env", self.env.close)
            self.env = None

        # 3. Create a fresh USD stage so the next gym.make() starts clean.
        _run_step("create new stage", sim_utils.create_new_stage)

        # 4. Synchronize CUDA to flush any pending work.
        _run_step("synchronize CUDA", torch.cuda.synchronize)

        # 5. Release shared buffers.
        self._obs_buffer = None
        self._rewards_buffer = None
        self._terminated_buffer = None
        self._truncated_buffer = None
        self._action_buffer = None

        # 7. Run garbage collection and release PyTorch's CUDA memory cache
        #    so freed GPU blocks are returned to the driver.
        _run_step("garbage collect", gc.collect)
        _run_step("empty CUDA cache", torch.cuda.empty_cache)

    def _cleanup_env(self):
        """Best-effort teardown that never raises and never touches the socket."""
        self._teardown_env(raise_on_error=False)

    def run(self):
        server_running = True
        while server_running:
            try:
                self._prepare_socket()
                print("[isaac_server] Waiting for connection…")
                self._wait_for_client()
                assert self._sock is not None
                print("[isaac_server] Client connected.")

                while True:
                    sig = self._sock.recv(1)
                    if not sig:
                        # Client closed the connection (EOF).
                        print("[isaac_server] Client disconnected.")
                        break
                    elif sig == self.MAKE:
                        self._make()
                    elif sig == self.STEP:
                        self._step()
                    elif sig == self.RESET:
                        self._reset()
                    elif sig == self.CLOSE:
                        self._close()
                    elif sig == self.STOP:
                        self._stop()
                        print("[isaac_server] Stop signal received. Shutting down.")
                        server_running = False
                        break
                    else:
                        raise ValueError(f"Unknown signal received: {sig}")
            except KeyboardInterrupt:
                # This block triggers ONLY when you press Ctrl+C
                print("\nKeyboardInterrupt caught! Gracefully shutting down...")
                server_running = False
            except Exception as e:
                print(f"[isaac_server] Error: {e}")
                # traceback.print_exc()
            finally:
                self._cleanup_env()

                if self._sock is not None:
                    self._sock.close()
                    self._sock = None
                if self.srv is not None:
                    self.srv.close()
                    self.srv = None
                print("[isaac_server] Connection closed.")

        print("[isaac_server] Server stopped.")