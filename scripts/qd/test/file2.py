#!/home/pablo/Documents/test/venv2/bin/python3
import socket

SOCKET_PATH = "/tmp/sockets/test_comm.sock"

client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.connect(SOCKET_PATH)

while True:
    data = client.recv(1)
    if not data:
        break
    if data == b'\x01':
        print("Received 0x01")
        choice = input("Type 'a' or 'b': ").strip()
        if choice == 'a':
            client.send(b'\x02')
            print("Sent 0x02")
        elif choice == 'b':
            client.send(b'\x03')
            print("Sent 0x03, exiting.")
            break

client.close()
