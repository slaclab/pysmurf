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
    data, addr = sock.recvfrom(64000) # buffer size is 1024 bytes
    d = json.loads(data)
    print(d)
    if last_seq is None:
        print('New sequence: %i' % d['seq_no'])
    else:
        delta = d['seq_no'] - last_seq
        if delta != 1:
            print('Sequence jump: %i + %i' % (last_seq, delta))
            types = []
    last_seq = d['seq_no']
    if not d['type'] in types:
        print('New type: %s at %i' % (d['type'], d['seq_no']))
        types.append(d['type'])

