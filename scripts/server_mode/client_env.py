import pickle
import socket
import struct
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

SOCKET_PATH = "/tmp/sockets/isaac_communication.sock"

STEP = b"\x01"
RESET = b"\x02"
CLOSE = b"\x03"

def receive_tensor(sock: socket.socket) -> torch.Tensor:
    """Helper function to receive a CUDA tensor via IPC."""
    (length,) = struct.unpack("!I", sock.recv(4))
    meta = pickle.loads(sock.recv(length))
    tensor = rebuild_cuda_tensor(**meta)
    return tensor

def main():
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(SOCKET_PATH)
    print("[client_env] Connected to isaac server env.")

    obs_buffer = None
    rewards_buffer = None
    terminated_buffer = None
    truncated_buffer = None
    action_buffer = None

    try:
        torch.cuda.init()

        # Receive IPC metadata for buffers
        obs_buffer = receive_tensor(sock)
        print(f"[client_env] Received obs_buffer with shape {obs_buffer.shape} and dtype {obs_buffer.dtype}")
        rewards_buffer = receive_tensor(sock)
        print(f"[client_env] Received rewards_buffer with shape {rewards_buffer.shape} and dtype {rewards_buffer.dtype}")
        terminated_buffer = receive_tensor(sock)
        print(f"[client_env] Received terminated_buffer with shape {terminated_buffer.shape} and dtype {terminated_buffer.dtype}")
        truncated_buffer = receive_tensor(sock)
        print(f"[client_env] Received truncated_buffer with shape {truncated_buffer.shape} and dtype {truncated_buffer.dtype}")
        action_buffer = receive_tensor(sock)
        print(f"[client_env] Received action_buffer with shape {action_buffer.shape} and dtype {action_buffer.dtype}")

        # Main loop
        for episode in range(1):
            print(f"[client_env] Starting episode {episode+1}...")

            cumulative_rewards = torch.zeros(rewards_buffer.shape[0], device=rewards_buffer.device)
            dones = torch.zeros(terminated_buffer.shape[0], dtype=torch.bool, device=terminated_buffer.device)
            step_count = torch.zeros(terminated_buffer.shape[0], dtype=torch.int32, device=terminated_buffer.device)

            # Send reset signal
            sock.sendall(RESET)
            sig = sock.recv(1)
            if sig != RESET:
                print(f"[client_env] Expected RESET signal, got {sig}")
                break

            for step in range(300):

                actions = 2 * torch.rand(action_buffer.shape, device=action_buffer.device) - 1
                action_buffer.copy_(actions)
                torch.cuda.synchronize()
                sock.sendall(STEP)
                sig = sock.recv(1)
                if sig != STEP:
                    print(f"[client_env] Expected STEP signal, got {sig}")
                    break

                alive = ~dones
                cumulative_rewards += rewards_buffer * alive.float()
                step_count += alive.int()

                dones |= (terminated_buffer | truncated_buffer)

                # Print dones for debugging
                # print(f"[client_env] Step {step+1}: dones = {dones.cpu().numpy()}")

                if dones.all():
                    break

        sock.sendall(CLOSE)
        sig = sock.recv(1)
        if sig != CLOSE:
            print(f"[client_env] Expected CLOSE signal, got {sig}")
        else:
            print("[client_env] Environment closed successfully.")

    finally:
        torch.cuda.synchronize()
        del obs_buffer
        del rewards_buffer
        del terminated_buffer
        del truncated_buffer
        del action_buffer
        sock.close()
        print("[client_env] Done.")

def main_old():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()

if __name__ == "__main__":
    main()
    