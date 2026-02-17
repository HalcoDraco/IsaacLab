#!/home/pablo/Documents/test/venv1/bin/python3
import socket
import os
import time

SOCKET_PATH = "/tmp/sockets/test_comm.sock"
SOCKET_DIR = os.path.dirname(SOCKET_PATH)

if SOCKET_DIR:
    os.makedirs(SOCKET_DIR, exist_ok=True)
    os.chmod(SOCKET_DIR, 0o755)

if os.path.exists(SOCKET_PATH):
    os.unlink(SOCKET_PATH)

server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCKET_PATH)
os.chmod(SOCKET_PATH, 0o777)
server.listen(1)

print("Waiting for connection...")
conn, _ = server.accept()
print("Connected.")

conn.send(b'\x01')
print("Sent 0x01")

while True:
    data = conn.recv(1)
    if not data:
        break
    if data == b'\x03':
        print("Received 0x03, exiting.")
        break
    elif data == b'\x02':
        print("Received 0x02, sleeping 1s and sending 0x01 again.")
        time.sleep(1)
        conn.send(b'\x01')
        print("Sent 0x01")

conn.close()
server.close()
os.unlink(SOCKET_PATH)
