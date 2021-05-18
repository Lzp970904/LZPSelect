import os
import time
import cv2
import copy
from paddleocr import PaddleOCR


def detect(image):
    # PATH_IMG_IN = './in'
    # filename = os.path.join(PATH_IMG_IN, '1.png')
    # filename = 'FSRCNNp2x2.jpg'
    # image = cv2.imread("picture2.jpg")
    # cv2.imshow("a", image)
    image1 = copy.deepcopy(image[0:int(image.shape[0]/2), 0:int(image.shape[1]/2), :])
    # print(image.shape)
    # cv2.imshow("b", image1)
    # cv2.waitKey()
    ocr = PaddleOCR()  # need to run only once to download and load model into memory
    start = time.perf_counter()
    result = ocr.ocr(image1, rec=True)
    # 第一个坐标表示的是第几个识别的字，第二个表示的识别的字中的是字的坐标还是还是字的内容或者置信度，第三个坐标表示的是四个坐
    # 标中的第几个坐标，第四个坐标是是一个具体的坐标的x或者y
    # print(result)
    end = time.perf_counter()
    print('检测文字区域 耗时{}'.format(end - start))
    # 每个矩形，从左上角顺时针排列
    for rect1 in result:
        if 'XODTC' in rect1[1][0]:
            # print(rect1[1][0])
            return rect1[1][0]
    #     cv2.line(image1, (int(rect1[0][0][0]), int(rect1[0][0][1])), (int(rect1[0][1][0]), int(rect1[0][1][1])), (0, 0, 255), 2)
    #     cv2.line(image1, (int(rect1[0][1][0]), int(rect1[0][1][1])), (int(rect1[0][2][0]), int(rect1[0][2][1])), (0, 0, 255), 2)
    #     cv2.line(image1, (int(rect1[0][2][0]), int(rect1[0][2][1])), (int(rect1[0][3][0]), int(rect1[0][3][1])), (0, 0, 255), 2)
    #     cv2.line(image1, (int(rect1[0][3][0]), int(rect1[0][3][1])), (int(rect1[0][0][0]), int(rect1[0][0][1])), (0, 0, 255), 2)
    # cv2.imshow("a", image1)
    # cv2.waitKey()
