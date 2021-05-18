import socket
import json
import numpy as np
import cv2


def recvall(sock, count):
    buf = b''  # buf是一个byte类型
    while count:
        # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 连接服务端
s.connect(('127.0.0.1', 6667))
# 向服务端发送数据
# b = ['hello', 'world']
# data = json.dumps(b)
# s.sendall(bytes(data.encode('utf-8')))
# # 接受服务端响应数据
while True:
    receive = s.recv(16)
    print("接收到")
    if len(receive):
        # print(str(receive, encoding='utf-8'))  ### 之前接受的帧率数据，现在换成image流数据
        # stringData = recvall(sock.recv(int(receive)), int(receive))
        stringData = recvall(s, int(receive))
        data = np.frombuffer(stringData, np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        cv2.imshow('show', image)
        cv2.waitKey(1)
s.close()