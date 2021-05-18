import socket
import json
import sys
import cv2
import numpy as np

address = ('127.0.0.1', 6666)
try:
    # 建立socket对象，参数意义见https://blog.csdn.net/rebelqsp/article/details/22109925
    # socket.AF_INET：服务器之间网络通信
    # socket.SOCK_STREAM：流式socket , for TCP
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 开启连接
    sock.connect(address)
except socket.error as msg:
    print(msg)
    sys.exit(1)

# # 读取一帧图像，读取成功:ret=1 frame=读取到的一帧图像；读取失败:ret=0
# ret, frame = capture.read()
# 压缩参数，后面cv2.imencode将会用到，对于jpeg来说，15代表图像质量，越高代表图像质量越好为 0-100，默认95
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    # 建立矩阵
    data = np.array(imgencode)
    # 将numpy矩阵转换成字符形式，以便在网络中传输
    stringData = data.tostring()

    # 先发送要发送的数据的长度
    # ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串
    sock.send(str.encode(str(len(stringData)).ljust(16)))
    # 发送数据
    sock.send(stringData)
    print("fasong end")