#!/usr/bin/env python3
"""
GPU File 2 (client) — runs in container B.

Connects to file1, reconstructs the shared GPU buffer via CUDA IPC,
divides it by 2 each iteration.

Requires: docker run --gpus all --ipc=host -v /tmp/sockets:/tmp/sockets
"""
import pickle
import socket
import struct
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

SOCKET_PATH = "/tmp/sockets/gpu_test_comm.sock"
CONT, STOP = b"\x01", b"\x02"

def main():

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(SOCKET_PATH)
    print("[file2] Connected.")

    buf = None
    try:
        # Receive IPC metadata
        torch.cuda.init()
        (length,) = struct.unpack("!I", sock.recv(4))
        meta = pickle.loads(sock.recv(length))
        meta = dict(meta)  # Ensure it's a regular dict for easier access

        buf = rebuild_cuda_tensor(**meta)
        sig = sock.recv(1)
        print(buf[0, :5])  # Debug: print first 5 elements of the first row
        sum_tensor = buf.clone()
        torch.cuda.synchronize()
        sock.sendall(CONT)

        # Main loop
        it = 0
        while True:
            sig = sock.recv(1)
            if sig != CONT:
                break
            it += 1
            buf += sum_tensor 
            torch.cuda.synchronize()
            sock.sendall(CONT)
        print(buf[0, :5])
    finally:
        if buf is not None:
            torch.cuda.synchronize()
            del buf
        sock.close()
        print("[file2] Done.")


if __name__ == "__main__":
    main()
