#!/usr/bin/env python3
"""
GPU File 1 (server) — runs in container A.

Zero-copy CUDA IPC between Docker containers:
  1. Allocates a GPU buffer, shares its IPC handle via Unix socket.
  2. Fills the buffer with random values, signals file2.
  3. Loop: file2 divides by 2 → if sum > threshold: exit, else: ×3, continue.

Requires: docker run --gpus all --ipc=host -v /tmp/sockets:/tmp/sockets
"""
import os
import pickle
import socket
import struct
import torch
import time

SOCKET_PATH = "/tmp/sockets/gpu_test_comm.sock"
SHAPE = (8192, 64)
DTYPE = torch.float32
THRESHOLD = 10000000.0
CONT, STOP = b"\x01", b"\x02"

def main():
    os.makedirs(os.path.dirname(SOCKET_PATH), exist_ok=True)
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCKET_PATH)
    srv.listen(1)
    print("[file1] Waiting for connection…")
    conn, _ = srv.accept()
    print("[file1] Connected.")

    buf = None
    try:
        buf = torch.empty(SHAPE, dtype=DTYPE, device="cuda")

        # Share buffer via CUDA IPC
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
        meta = pickle.dumps({
            "tensor_cls": torch.Tensor,
            "tensor_size": buf.shape,
            "tensor_stride": buf.stride(),
            "tensor_offset": buf.storage_offset(),
            "storage_cls": torch.UntypedStorage,
            "dtype": buf.dtype,
            "requires_grad": False,
            **dict(zip(ipc_keys, buf.untyped_storage()._share_cuda_())),
        })
        conn.sendall(struct.pack("!I", len(meta)) + meta)
        print("[file1] Sent IPC handle.")

        # Fill buffer with random data, signal file2
        random_data = torch.rand(SHAPE, dtype=DTYPE, device="cuda")
        random_data.div_(10)  # Start with smaller values to allow more iterations before hitting the threshold
        buf.copy_(random_data)
        sum_tensor = buf.clone() 
        torch.cuda.synchronize()
        print(buf[0, :5])  # Debug: print first 5 elements of the first row
        conn.sendall(CONT)
        conn.recv(1)  # Wait for file2 to acknowledge before starting the loop

        # Main loop
        # it = 0
        start_time = time.perf_counter()
        for it in range(1, 100):
            # if conn.recv(1) != CONT:
            #     break
            # it += 1
            # torch.cuda.synchronize()
            # s = buf.sum().item()
            # print(f"Iteration {it} starting...")
            buf += sum_tensor
            torch.cuda.synchronize()
            conn.sendall(CONT)
            # if it % 150 == 0:
            #     elapsed = time.perf_counter() - start_time
            #     print(f"it/sec: {150/elapsed:.2f}")
            #     start_time = time.perf_counter()
            conn.recv(1)  # Wait for file2 to acknowledge the last iteration
        conn.sendall(STOP)
        print(buf[0, :5])
    finally:
        if buf is not None:
            torch.cuda.synchronize()
            del buf
        conn.close()
        srv.close()
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)
        # Remove /tmp/sockets directory if empty

        # os.rmdir(os.path.dirname(SOCKET_PATH))

        print("[file1] Done.")


if __name__ == "__main__":
    main()
