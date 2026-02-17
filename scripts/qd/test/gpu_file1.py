#!/home/pablo/Documents/test/venv1/bin/python3
"""
GPU File 1 (server) — runs as root in production, uses venv1.

Zero-copy CUDA IPC test:
  1. Preallocates a GPU buffer and shares its IPC handle with file2.
  2. Fills the buffer with random values, signals file2.
  3. After file2 divides by 2, checks sum > THRESHOLD_N:
       - yes → sends 0x02 (exit)
       - no  → multiplies by 3 in-place, sends 0x01 (continue)
"""
import socket
import os
import pickle
import struct
import torch

# ── Configuration ──────────────────────────────────────────────────────────────
SOCKET_PATH = "/tmp/sockets/gpu_test_comm.sock"  # use /tmp/gpu_test_comm.sock for non-root testing
TENSOR_SHAPE = (100, 100)
DTYPE = torch.float32
THRESHOLD_N = 5000.0
# ───────────────────────────────────────────────────────────────────────────────

# ── Helpers ────────────────────────────────────────────────────────────────────
def recv_exact(sock, n):
    """Receive exactly *n* bytes from *sock*."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def send_msg(sock, data: bytes):
    """Send a length-prefixed message."""
    sock.sendall(struct.pack("!I", len(data)) + data)
# ───────────────────────────────────────────────────────────────────────────────

# Permissive umask so CUDA IPC shared-memory files are accessible by non-root
old_umask = os.umask(0o000)

# ── Socket setup ───────────────────────────────────────────────────────────────
SOCKET_DIR = os.path.dirname(SOCKET_PATH)
if SOCKET_DIR:
    os.makedirs(SOCKET_DIR, exist_ok=True)
    try:
        os.chmod(SOCKET_DIR, 0o755)
    except PermissionError:
        pass  # directory already exists with adequate perms

if os.path.exists(SOCKET_PATH):
    os.unlink(SOCKET_PATH)

server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCKET_PATH)
os.chmod(SOCKET_PATH, 0o777)
server.listen(1)

print("[file1] Waiting for connection...")
conn, _ = server.accept()
print("[file1] Connected.")

# ── Step 1: pre-allocate GPU buffer ────────────────────────────────────────────
buffer = torch.zeros(TENSOR_SHAPE, dtype=DTYPE, device="cuda")
print(f"[file1] Pre-allocated buffer {TENSOR_SHAPE} on GPU")

# ── Share buffer via CUDA IPC ──────────────────────────────────────────────────
storage = buffer.untyped_storage()
(
    storage_device,
    storage_handle,
    storage_size_bytes,
    storage_offset_bytes,
    ref_counter_handle,
    ref_counter_offset,
    event_handle,
    event_sync_required,
) = storage._share_cuda_()

ipc_meta = {
    "tensor_size": tuple(buffer.shape),
    "tensor_stride": tuple(buffer.stride()),
    "tensor_offset": buffer.storage_offset(),
    "dtype": str(buffer.dtype),           # send as string for cross-version compat
    "storage_device": storage_device,
    "storage_handle": storage_handle,
    "storage_size_bytes": storage_size_bytes,
    "storage_offset_bytes": storage_offset_bytes,
    "ref_counter_handle": ref_counter_handle,
    "ref_counter_offset": ref_counter_offset,
    "event_handle": event_handle,
    "event_sync_required": event_sync_required,
}

send_msg(conn, pickle.dumps(ipc_meta))
print("[file1] Sent IPC handle to file2")

# ── Steps 2-3: fill buffer with random data ────────────────────────────────────
random_tensor = torch.rand(TENSOR_SHAPE, dtype=DTYPE, device="cuda")
buffer.copy_(random_tensor)
torch.cuda.synchronize()
print(f"[file1] Copied random tensor to buffer  (sum = {buffer.sum().item():.2f})")

# ── Step 4: signal file2 that the buffer is ready ─────────────────────────────
conn.send(b"\x01")
print("[file1] Sent 0x01 (buffer ready)\n")

# ── Main loop ──────────────────────────────────────────────────────────────────
iteration = 0
while True:
    # Step 7: wait for file2's "done" flag
    data = conn.recv(1)
    if not data:
        print("[file1] Connection lost")
        break

    if data == b"\x01":
        iteration += 1
        torch.cuda.synchronize()
        current_sum = buffer.sum().item()
        print(f"[file1] Iteration {iteration}: file2 done  →  buffer sum = {current_sum:.2f}")

        # Step 8
        if current_sum > THRESHOLD_N:
            print(f"[file1] {current_sum:.2f} > {THRESHOLD_N}  →  sending 0x02 (exit)")
            conn.send(b"\x02")
            break
        else:
            buffer.mul_(3)
            torch.cuda.synchronize()
            new_sum = buffer.sum().item()
            print(f"[file1] {current_sum:.2f} <= {THRESHOLD_N}  →  buffer *= 3  (sum = {new_sum:.2f})")
            # Step 9
            conn.send(b"\x01")
            print("[file1] Sent 0x01 (continue)\n")

# ── Cleanup ────────────────────────────────────────────────────────────────────
print("\n[file1] Exiting.")
os.umask(old_umask)
conn.close()
server.close()
if os.path.exists(SOCKET_PATH):
    os.unlink(SOCKET_PATH)
