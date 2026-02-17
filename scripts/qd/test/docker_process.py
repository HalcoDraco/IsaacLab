import torch
import socket
import pickle
import os
import time

# === CONFIG ===
N, M = 4, 8
DOCK_HOST_PATH = '/tmp/sockets/docker_to_host.sock'
HOST_DOCK_PATH = '/tmp/sockets/host_to_docker.sock'
HANDLE_PATH = '/tmp/sockets/docker_handle.pkl'

# === ALLOCATE BUFFER ===
buf = torch.zeros(N, M, dtype=torch.float32, device='cuda')

# === EXPORT AND SAVE HANDLE ===
handle_info = buf.untyped_storage()._share_cuda_()  # Get the IPC handle info
os.makedirs(os.path.dirname(HANDLE_PATH), exist_ok=True)
with open(HANDLE_PATH, 'wb') as f:
    pickle.dump({'handle': handle_info, 'shape': (N, M), 'dtype': torch.float32}, f)
os.chmod(HANDLE_PATH, 0o666)  # Allow host (non-root) to read/delete

# === SETUP SOCKETS ===
if os.path.exists(DOCK_HOST_PATH): os.unlink(DOCK_HOST_PATH)
sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
sock.bind(DOCK_HOST_PATH)
os.chmod(DOCK_HOST_PATH, 0o666)  # Allow host (non-root) to sendto this path

print("Docker process is ready and waiting for the host...")

# Wait for host socket to appear
while not os.path.exists(HOST_DOCK_PATH):
    time.sleep(0.05)

print("Docker process connected to host.")

self_tensor = torch.arange(N * M, dtype=torch.float32, device='cuda').reshape(N, M)

# === LOOP ===
for i in range(100):
    # Write something into the buffer
    buf.copy_(self_tensor)
    torch.cuda.synchronize()  # Ensure the data is written before sending flag
    print(f"Docker wrote:\n{buf}")

    flag_sent = b'\x01'

    # Send flag to host (host is bound to HOST_DOCK_PATH)
    sock.sendto(flag_sent, HOST_DOCK_PATH)

    print("Docker sent flag x01 to host, waiting for host to read...")

    # Receive on our own bound socket (host sends to DOCK_HOST_PATH)
    flag = sock.recv(1)

    if flag == b'\x02':
        print("Docker received flag x02 from host.")
    elif flag == b'\x03':
        print("Docker received flag x03 from host, exiting loop.")
        break

    print(f"Docker read back:\n{buf}")

    self_tensor *= 3  # Update tensor for next iteration

    time.sleep(1)

os.remove(HANDLE_PATH)
print("Docker process exiting.")