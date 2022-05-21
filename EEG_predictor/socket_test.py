import socket
import numpy as np
import json
a= str(np.zeros((12, 521, 1090)))
print(len(a))
sk = socket.socket()
sk.connect(('192.168.1.100',30002))
while True:
    info =json.dumps(str(np.zeros((12,521,1090))))
    sk.send(bytes(info,encoding='utf-8'))
    #信息接受
    ret = sk.recv(1024)
    if ret == b'bye':
        sk.send(b'bye')
        break
    print(ret.decode('utf-8'))
sk.close()