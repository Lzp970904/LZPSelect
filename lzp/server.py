import socket
from paddleocr import PaddleOCR
import numpy as np
import cv2
from queue import LifoQueue
import threading


def recvall(sock, count):
    buf = b''  # buf是一个byte类型
    while count:
        # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def receive():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 绑定地址
    s.bind(('', 6666))
    # 监听连接
    s.listen(5)
    print("等待")
    conn, addr = s.accept()
    print('connect from:' + str(addr))
    while True:
        length = recvall(conn, 16)  # 获得图片文件的长度,16代表获取长度
        stringData = recvall(conn, int(length))  # 根据获得的文件长度，获取图片文件
        data = np.frombuffer(stringData, np.uint8)  # 将获取到的字符流数据转换成1维数组
        decimg = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 将数组解码成图像
        print("接收完成")
        q.put(decimg)


def solve():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 绑定地址
    s.bind(('', 6667))
    # 监听连接
    s.listen(5)
    print("等待")
    conn, addr = s.accept()
    print('connect from:' + str(addr))
    while True:
        image = q.get()
        result = ocr.ocr(image, rec=True)
        print(result)
        for rect1 in result:
            text = rect1[1][0]
            cv2.putText(image, text, (int(rect1[0][0][0]), int(rect1[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
            cv2.line(image, (int(rect1[0][0][0]), int(rect1[0][0][1])),
                     (int(rect1[0][1][0]), int(rect1[0][1][1])),
                     (0, 0, 255), 2)
            cv2.line(image, (int(rect1[0][1][0]), int(rect1[0][1][1])),
                     (int(rect1[0][2][0]), int(rect1[0][2][1])),
                     (0, 0, 255), 2)
            cv2.line(image, (int(rect1[0][2][0]), int(rect1[0][2][1])),
                     (int(rect1[0][3][0]), int(rect1[0][3][1])),
                     (0, 0, 255), 2)
            cv2.line(image, (int(rect1[0][3][0]), int(rect1[0][3][1])),
                     (int(rect1[0][0][0]), int(rect1[0][0][1])),
                     (0, 0, 255), 2)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        result, imgencode = cv2.imencode('.jpg', image, encode_param)
        # 建立矩阵
        data = np.array(imgencode)
        # 将numpy矩阵转换成字符形式，以便在网络中传输
        img_Data = data.tostring()

        # 先发送要发送的数据的长度
        # ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串
        print("处理完成，开始回传")
        conn.send(str.encode(str(len(img_Data)).ljust(16)))
        # # print(img_Data)
        # # 发送数据
        conn.send(img_Data)
        print("回传完成")


ocr = PaddleOCR()
q = LifoQueue(maxsize=0)
thread1 = threading.Thread(target=receive)
thread2 = threading.Thread(target=solve)
thread1.start()
thread2.start()
thread1.join()
thread2.join()
