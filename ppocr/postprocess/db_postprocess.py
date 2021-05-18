# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.fluid as fluid

import numpy as np
import string
import cv2
from shapely.geometry import Polygon
import pyclipper
from copy import deepcopy
import math


class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self, params):
        self.thresh = params['thresh']
        self.box_thresh = params['box_thresh']
        self.max_candidates = params['max_candidates']
        self.unclip_ratio = params['unclip_ratio']
        self.min_size = 3
        self.dilation_kernel = np.array([[1, 1], [1, 1]])

    def boxes_from_bitmap(self, pred, mask):
        """
        Get boxes from the binarized image predicted by DB.
        :param pred: the binarized image predicted by DB.
        :param mask: new 'pred' after threshold filtering.
        :return: (boxes, the score of each boxes)
        """
        lzp = cv2.imread("img_464.jpg")
        lzp = cv2.resize(lzp, (960, 512))
        # cv2.imshow("pred", pred)
        dest_height, dest_width = pred.shape[-2:]
        bitmap = deepcopy(mask)
        height, width = bitmap.shape
        # cv2.imshow("bitmap", bitmap * 255)
        # cv2.waitKey()
        # 这里是根据bitmap找寻轮廓
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours, ), dtype=np.float32)

        for index in range(num_contours):
            # 这里num_contours是bitmap的轮廓，threshold之后的二值图像
            contour = contours[index]
            # 此处返回的是最小面积的外界矩形的长度和高度较小的那个
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            # print("first {}".format(sside))
            points = np.array(points)
            # 这里将pred也带入，points是之前threshold的一个轮廓的最小外界矩形
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            # cv2.line(lzp, (int(points[0][0]), int(points[0][1])), (int(points[1][0]),int(points[1][1])), (0, 0, 255), 2)
            # cv2.line(lzp, (int(points[1][0]), int(points[1][1])), (int(points[2][0]),int(points[2][1])), (0, 0, 255), 2)
            # cv2.line(lzp, (int(points[2][0]), int(points[2][1])), (int(points[3][0]),int(points[3][1])), (0, 0, 255), 2)
            # cv2.line(lzp, (int(points[3][0]), int(points[3][1])), (int(points[0][0]),int(points[0][1])), (0, 0, 255), 2)
            # lzpx = math.sqrt((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)
            # lzpy = math.sqrt((points[1][0] - points[2][0])**2 + (points[1][1] - points[2][1])**2)
            # print("points[0][0] = {}".format(points[0][0]))
            # print("points[0][1] = {}".format(points[0][1]))
            # print("points[1][0] = {}".format(points[1][0]))
            # print("points[1][1] = {}".format(points[1][1]))
            # print("points[2][0] = {}".format(points[2][0]))
            # print("points[2][1] = {}".format(points[2][1]))
            # print("x = {}".format(lzpx))
            # print("y = {}".format(lzpy))
            # lzpbl = lzpx/lzpy
            # print("bilv = {}".format(lzpbl))
            # cv2.imshow("lzp", lzp)
            # cv2.waitKey()
            # print("index:{} + score:{}".format(index, score))
            if self.box_thresh > score:
                continue
            # 这里计算的是threshold的最小的矩形，将其扩张或者收缩
            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            # print("second {}".format(sside))
            # print("----------------")
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            # cv2.line(lzp, (int(box[0][0]), int(box[0][1])), (int(box[1][0]),int(box[1][1])), (0, 0, 255), 2)
            # cv2.line(lzp, (int(box[1][0]), int(box[1][1])), (int(box[2][0]),int(box[2][1])), (0, 0, 255), 2)
            # cv2.line(lzp, (int(box[2][0]), int(box[2][1])), (int(box[3][0]),int(box[3][1])), (0, 0, 255), 2)
            # cv2.line(lzp, (int(box[3][0]), int(box[3][1])), (int(box[0][0]),int(box[0][1])), (0, 0, 255), 2)
            # cv2.imshow("lzp", lzp)
            # cv2.waitKey()
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def unclip(self, box):
        """
        Shrink or expand the boxaccording to 'unclip_ratio'
        :param box: The predicted box.
        :return: uncliped box
        """
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        """
        Get boxes from the contour or box.
        :param contour: The predicted contour.
        :return: The predicted box.
        """
        # lzp = cv2.imread("img_5.jpg")
        bounding_box = cv2.minAreaRect(contour)
        # lzpbox = cv2.boxPoints(bounding_box)
        #
        # lzpbox = np.int0(lzpbox)
        # # 画出来
        # cv2.drawContours(lzp, [lzpbox], 0, (0, 0, 255), 3)
        # cv2.imshow("lzp", lzp)
        # cv2.waitKey()

        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        """
        Calculate the score of box.
        :param bitmap: The binarized image predicted by DB.
        :param _box: The predicted box
        :return: score
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        # np.floor对数据取整
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        # cv2.fillPoly填充图形，填充该区域为指定颜色
        # 比如：cv::Scalar
        # mean = cv::mean(image, mask);
        # mask是与iamge一样大小的矩阵，其中的数值为0或者1，为1的地方，计算出image中所有元素的均值，为0
        # 的地方，不计算
        # 换算成三维为1，两列，再用总数据除以2，以mask为底板，在mask上的box的区域一里面进行1的填充
        # mask因为为向上取整后，会比box大一些

        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        # print(cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask))
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, ratio_list):
        pred = outs_dict['maps']

        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh
        boxes_batch = []
        for batch_index in range(pred.shape[0]):

            mask = cv2.dilate(
                np.array(segmentation[batch_index]).astype(np.uint8),
                self.dilation_kernel)
            tmp_boxes, tmp_scores = self.boxes_from_bitmap(pred[batch_index],
                                                           mask)

            boxes = []
            for k in range(len(tmp_boxes)):
                if tmp_scores[k] > self.box_thresh:
                    boxes.append(tmp_boxes[k])
            if len(boxes) > 0:
                boxes = np.array(boxes)

                ratio_h, ratio_w = ratio_list[batch_index]
                boxes[:, :, 0] = boxes[:, :, 0] / ratio_w
                boxes[:, :, 1] = boxes[:, :, 1] / ratio_h

            boxes_batch.append(boxes)
        return boxes_batch
