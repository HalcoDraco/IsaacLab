# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import os
import pickle
import socket
import struct

SOCKET_PATH = "/tmp/sockets/isaac_communication.sock"

STEP = b"\x01"
RESET = b"\x02"
CLOSE = b"\x03"

def get_tensor_metadata(tensor: torch.Tensor) -> dict:
    """Extract metadata from a CUDA tensor for IPC sharing."""
    if not tensor.is_cuda:
        raise ValueError("Only CUDA tensors can be shared via IPC.")
    
    ipc_keys = [
        "storage_device", 
        "storage_handle", 
        "storage_size_bytes",
        "storage_offset_bytes", 
        "ref_counter_handle", 
        "ref_counter_offset",
        "event_handle", 
        "event_sync_required",
    ]
    meta = {
        "tensor_cls": type(tensor),
        "tensor_size": tensor.shape,
        "tensor_stride": tensor.stride(),
        "tensor_offset": tensor.storage_offset(),
        "storage_cls": torch.UntypedStorage,
        "dtype": tensor.dtype,
        "requires_grad": tensor.requires_grad,
        **dict(zip(ipc_keys, tensor.untyped_storage()._share_cuda_())),
    }
    return meta

def main():
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Create buffers for IPC sharing
    obs_buffer = torch.empty(env.observation_space.shape, device=env.unwrapped.device)
    rewards_buffer = torch.empty((args_cli.num_envs,), device=env.unwrapped.device)
    terminated_buffer = torch.empty((args_cli.num_envs,), dtype=torch.bool, device=env.unwrapped.device)
    truncated_buffer = torch.empty((args_cli.num_envs,), dtype=torch.bool, device=env.unwrapped.device)
    action_buffer = torch.empty(env.action_space.shape, device=env.unwrapped.device)

    os.makedirs(os.path.dirname(SOCKET_PATH), exist_ok=True)
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCKET_PATH)
    srv.listen(1)
    print("[isaac_server] Waiting for connection…")
    conn, _ = srv.accept()
    print("[isaac_server] Client connected.")

    # Send IPC metadata for buffers
    obs_meta = pickle.dumps(get_tensor_metadata(obs_buffer))
    rewards_meta = pickle.dumps(get_tensor_metadata(rewards_buffer))
    terminated_meta = pickle.dumps(get_tensor_metadata(terminated_buffer))
    truncated_meta = pickle.dumps(get_tensor_metadata(truncated_buffer))
    action_meta = pickle.dumps(get_tensor_metadata(action_buffer))

    conn.sendall(struct.pack("!I", len(obs_meta)) + obs_meta)
    conn.sendall(struct.pack("!I", len(rewards_meta)) + rewards_meta)
    conn.sendall(struct.pack("!I", len(terminated_meta)) + terminated_meta)
    conn.sendall(struct.pack("!I", len(truncated_meta)) + truncated_meta)
    conn.sendall(struct.pack("!I", len(action_meta)) + action_meta)
    print("[isaac_server] Sent IPC metadata for buffers.")

    try:
        # Main loop
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # Wait for client signal
                sig = conn.recv(1)
                if sig == STEP:
                    obs_dict, rewards, terminated, truncated, infos = env.step(action_buffer)
                    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
                    obs_buffer.copy_(obs)
                    rewards_buffer.copy_(rewards)
                    terminated_buffer.copy_(terminated)
                    truncated_buffer.copy_(truncated)
                    torch.cuda.synchronize()
                    conn.sendall(STEP)
                elif sig == RESET:
                    obs_dict, _ = env.reset()
                    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
                    obs_buffer.copy_(obs)
                    torch.cuda.synchronize()
                    conn.sendall(RESET)
                elif sig == CLOSE:
                    env.close()
                    torch.cuda.synchronize()
                    conn.sendall(CLOSE)
                    print("[isaac_server] Environment closed by client.")
                    break
                else:
                    print(f"[isaac_server] Received unknown signal: {sig}")
    finally:
        torch.cuda.synchronize()
        del obs_buffer
        del rewards_buffer
        del terminated_buffer
        del truncated_buffer
        del action_buffer
        conn.close()
        srv.close()
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)
        print("[isaac_server] Done.")

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
