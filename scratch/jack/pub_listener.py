import socket
import json

UDP_IP = "localhost"
UDP_PORT = 8200

sock = socket.socket(socket.AF_INET,
                     socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

last_seq = None
types = []

while True:
    _data, addr = sock.recvfrom(64000) # buffer size is 1024 bytes
    data = json.loads(_data)

    print(data)