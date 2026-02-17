#!/home/pablo/Documents/test/venv2/bin/python3
"""
GPU File 2 (client) — uses venv2.

Zero-copy CUDA IPC test:
  1. Connects to file1 and receives the CUDA IPC handle.
  2. Reconstructs the shared GPU buffer (zero-copy).
  3. On receiving 0x01: divides the buffer by 2 in-place, signals back.
  4. On receiving 0x02: exits.
"""
import socket
import pickle
import struct
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

# ── Configuration ──────────────────────────────────────────────────────────────
SOCKET_PATH = "/tmp/sockets/gpu_test_comm.sock"  # use /tmp/gpu_test_comm.sock for non-root testing
# ───────────────────────────────────────────────────────────────────────────────

DTYPE_MAP = {
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float16": torch.float16,
    "torch.int32":   torch.int32,
    "torch.int64":   torch.int64,
}

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


def recv_msg(sock):
    """Receive a length-prefixed message."""
    raw = recv_exact(sock, 4)
    if not raw:
        return None
    length = struct.unpack("!I", raw)[0]
    return recv_exact(sock, length)
# ───────────────────────────────────────────────────────────────────────────────

# ── Connect to file1 ──────────────────────────────────────────────────────────
client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.connect(SOCKET_PATH)
print("[file2] Connected to file1.")

# ── Receive IPC metadata and reconstruct the shared buffer ─────────────────────
raw_meta = recv_msg(client)
ipc_meta = pickle.loads(raw_meta)
print(f"[file2] Received IPC metadata  (shape = {ipc_meta['tensor_size']})")

dtype = DTYPE_MAP.get(ipc_meta["dtype"], torch.float32)

buffer = rebuild_cuda_tensor(
    tensor_cls=torch.Tensor,
    tensor_size=torch.Size(ipc_meta["tensor_size"]),
    tensor_stride=ipc_meta["tensor_stride"],
    tensor_offset=ipc_meta["tensor_offset"],
    storage_cls=torch.UntypedStorage,
    dtype=dtype,
    storage_device=ipc_meta["storage_device"],
    storage_handle=ipc_meta["storage_handle"],
    storage_size_bytes=ipc_meta["storage_size_bytes"],
    storage_offset_bytes=ipc_meta["storage_offset_bytes"],
    requires_grad=False,
    ref_counter_handle=ipc_meta["ref_counter_handle"],
    ref_counter_offset=ipc_meta["ref_counter_offset"],
    event_handle=ipc_meta["event_handle"],
    event_sync_required=ipc_meta["event_sync_required"],
)
print(f"[file2] Reconstructed shared GPU buffer {buffer.shape}\n")

# ── Main loop ──────────────────────────────────────────────────────────────────
iteration = 0
while True:
    data = client.recv(1)
    if not data:
        print("[file2] Connection lost")
        break

    if data == b"\x01":
        iteration += 1
        # Step 5: divide the shared buffer by 2 in-place
        torch.cuda.synchronize()
        pre_sum = buffer.sum().item()
        buffer.div_(2)
        torch.cuda.synchronize()
        post_sum = buffer.sum().item()
        print(f"[file2] Iteration {iteration}: buffer /= 2  ({pre_sum:.2f} → {post_sum:.2f})")

        # Step 6: signal file1 that processing is done
        client.send(b"\x01")

    elif data == b"\x02":
        # Step 10: exit
        print(f"\n[file2] Received 0x02 → exiting.")
        break

# ── Cleanup ────────────────────────────────────────────────────────────────────
print("[file2] Exiting.")
client.close()
