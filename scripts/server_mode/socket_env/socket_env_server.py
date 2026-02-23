import os
import socket
import struct
import torch
import gymnasium as gym

import isaaclab.sim as sim_utils
import omni.physx
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from .socket_env_base import SocketEnv

class SocketEnvServer(SocketEnv):

    def __init__(self, socket_path=None, disable_fabric: bool = False):
        super().__init__(socket_path, disable_fabric=disable_fabric)
        self.env: gym.Env | None = None

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
                self.sock, _ = self.srv.accept()
                break
            except socket.timeout:
                continue

    def _send_tensor_metadata(self, tensor: torch.Tensor):
        if self.sock is None:
            raise RuntimeError("No client connected.")
        self._send_pickled_object(self._get_tensor_metadata(tensor))

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
        if self.sock is None:
            raise RuntimeError("No client connected.")
        
        if self.env is not None:
            self._cleanup_env()

        (task_len,) = struct.unpack("!I", self._recv_exact(4))
        task = self._recv_exact(task_len).decode("utf-8")
        (num_envs,) = struct.unpack("!i", self._recv_exact(4))

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
        self._send_pickled_object(self.env.observation_space)
        self._send_pickled_object(self.env.action_space)

    def _step(self):
        if self.env is None:
            raise RuntimeError("Environment not initialized.")
        if self.sock is None:
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
        self.sock.sendall(self.STEP)

    def _reset(self):
        if self.env is None:
            raise RuntimeError("Environment not initialized.")
        if self.sock is None:
            raise RuntimeError("No client connected.")
        if self.obs_buffer is None:
            raise RuntimeError("Observation buffer not initialized.")
        
        obs_dict, _ = self.env.reset()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        self.obs_buffer.copy_(obs)
        torch.cuda.synchronize()
        self.sock.sendall(self.RESET)

    def _close(self):
        if self.env is None:
            raise RuntimeError("Environment not initialized.")
        if self.sock is None:
            raise RuntimeError("No client connected.")
        
        # DirectRLEnv.close() only detaches the physx stage and stops the sim when
        # create_stage_in_memory is True (not the default). Without detaching first,
        # sim.stop() and create_new_stage() hang because the physics engine still
        # holds the stage. We must detach explicitly before closing.
        omni.physx.get_physx_simulation_interface().detach_stage()

        self.env.close()

        # Create a fresh USD stage so old prims don't interfere with the next gym.make()
        sim_utils.create_new_stage()
        torch.cuda.synchronize()

        self.env = None
        self.obs_buffer = None
        self.rewards_buffer = None
        self.terminated_buffer = None
        self.truncated_buffer = None
        self.action_buffer = None

        self.sock.sendall(self.CLOSE)

    def _stop(self):
        if self.sock is None:
            raise RuntimeError("No client connected.")
        self.sock.sendall(self.STOP)

    def _cleanup_env(self):
        """Tear down the current environment and leave the simulator ready for a new gym.make().

        Safe to call even when the environment is already closed or was never created.
        Mirrors the cleanup sequence of ``_close()`` (detach → env.close → new stage)
        but never raises and never touches the socket.
        """
        # 1. Detach physics so env.close() / create_new_stage() don't hang.
        try:
            omni.physx.get_physx_simulation_interface().detach_stage()
        except Exception as e:
            print(f"[isaac_server] Warning: Failed to detach stage: {e}")

        # 2. Close the Gym environment.
        if self.env is not None:
            try:
                self.env.close()
            except Exception as e:
                print(f"[isaac_server] Warning: Failed to close env: {e}")
            self.env = None

        # 3. Create a fresh USD stage so the next gym.make() starts clean.
        try:
            sim_utils.create_new_stage()
        except Exception as e:
            print(f"[isaac_server] Warning: Failed to create new stage: {e}")

        # 4. Synchronize CUDA to flush any pending work.
        try:
            torch.cuda.synchronize()
        except Exception as e:
            print(f"[isaac_server] Warning: Failed to synchronize CUDA: {e}")

        # 5. Release shared buffers.
        self.obs_buffer = None
        self.rewards_buffer = None
        self.terminated_buffer = None
        self.truncated_buffer = None
        self.action_buffer = None

    def run(self):
        server_running = True
        while server_running:
            try:
                self._prepare_socket()
                print("[isaac_server] Waiting for connection…")
                self._wait_for_client()
                assert self.sock is not None
                print("[isaac_server] Client connected.")

                while True:
                    sig = self.sock.recv(1)
                    if sig == self.MAKE:
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
            finally:
                self._cleanup_env()

                if self.sock is not None:
                    self.sock.close()
                    self.sock = None
                if self.srv is not None:
                    self.srv.close()
                    self.srv = None
                print("[isaac_server] Connection closed.")

        print("[isaac_server] Server stopped.")