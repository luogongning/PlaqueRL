from __future__ import division

import numbers
import os
import random

import numpy as np
import cv2

import itertools

from commons import minmax
from configs import ADNetConf
from networks import ADNetwork


class Coordinate:
    @staticmethod
    def get_imgwh(img):
        return Coordinate(x=img.shape[1], y=img.shape[0])

    def __init__(self, x, y):
        self.x = int(round(x))
        self.y = int(round(y))

    def __repr__(self):
        return 'x=%d, y=%d' % (self.x, self.y)

    def __add__(self, other):
        if isinstance(other, Coordinate):
            x = self.x + other.x
            y = self.y + other.y
        elif isinstance(other, numbers.Number):
            x = int(round(self.x + other))
            y = int(round(self.y + other))
        elif isinstance(other, tuple) or isinstance(other, list) or isinstance(other, np.ndarray):
            x = int(round(self.x + other[0]))
            y = int(round(self.y + other[1]))
        else:
            raise
        return Coordinate(x, y)

    def __sub__(self, other):
        return self.__add__(other * -1)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return Coordinate(self.x * other, self.y * other)
        elif isinstance(other, tuple):
            return Coordinate(self.x * other[0], self.y * other[1])
        raise

    def __floordiv__(self, other):
        return self.__truediv__(other)

    def __truediv__(self, other):
        if isinstance(other, Coordinate):
            x = self.x // other.x
            y = self.y // other.y
        elif isinstance(other, numbers.Number):
            x = self.x // other
            y = self.y // other
        else:
            raise
        return Coordinate(x, y)

    def __iter__(self):
        yield self.x
        yield self.y

    def __getitem__(self, item):
        if item == 0:
            return self.x
        else:
            return self.y

    def __eq__(self, other):
        if isinstance(other, Coordinate):
            return self.x == other.x and self.y == other.y
        elif isinstance(other, tuple) or isinstance(other, list) or isinstance(other, np.ndarray):
            return self.x == other[0] and self.y == other[1]
        else:
            raise

    def max(self, val):
        self.x = max(self.x, val)
        self.y = max(self.y, val)


class BoundingBox:
    COLOR_GT = (0, 255, 0)
    COLOR_PREDICT = (255, 0, 0)
    COLOR_NEGATIVE = (0, 0, 255)

    @staticmethod
    def read_vid_gt(path):
        if os.path.isdir(path):
            path = os.path.join(path, 'groundtruth_rect.txt')

        with open(path, 'r') as f:
            lines = f.readlines()

        boxes = []
        for line in lines:
            if not line.strip():
                continue
            x, y, w, h = [int(x) for x in line.split(',')]
            box = BoundingBox(x, y, w, h)
            boxes.append(box)
        return boxes

    @staticmethod
    def get_action_labels(samples, gt_box):
        # TODO : vectorize everything
        return [BoundingBox.get_action_label(sample, gt_box) for sample in samples]

    @staticmethod
    def get_action_label(sample, gt_box):
        ious = []
        for i in range(ADNetwork.NUM_ACTIONS):
            moved_box = sample.do_action(imgwh=None, action_idx=i)
            iou = gt_box.iou(moved_box)
            ious.append(iou)

        if ious[ADNetwork.ACTION_IDX_STOP] > ADNetConf.get()['predict']['stop_iou']:
            return ADNetwork.ACTION_IDX_STOP
        if max(ious[:-2]) * 0.99999 <= ious[ADNetwork.ACTION_IDX_STOP]:
            return np.argmax(ious)
            # return random.choice([i for i, x in enumerate(ious) if x >= max(ious)])
        return np.argmax(ious[:-2])
        # return random.choice([i for i, x in enumerate(ious[:-2]) if x >= max(ious[:-2])])

    def __init__(self, x, y, w, h):
        self.xy = Coordinate(x, y)
        self.wh = Coordinate(w, h)
        self.feat = None

    def __repr__(self):
        return 'x=%d, y=%d, w=%d, h=%d' % (self.xy.x, self.xy.y, self.wh.x, self.wh.y)

    def __eq__(self, other):
        return self.xy == other.xy and self.wh == other.wh

    def __add__(self, other):
        if isinstance(other, tuple) or isinstance(other, list) or isinstance(other, np.ndarray):
            xy = self.xy + other[:2]
            wh = self.wh + other[2:]
            return BoundingBox(xy.x, xy.y, wh.x, wh.y)
        elif isinstance(other, BoundingBox):
            xy = self.xy + other.xy
            wh = self.wh + other.wh
            return BoundingBox(xy.x, xy.y, wh.x, wh.y)
        raise

    def __mul__(self, other):
        if isinstance(other, tuple) or isinstance(other, list) or isinstance(other, np.ndarray):
            xy = self.xy * other[:2]
            wh = self.wh * other[2:]
            return BoundingBox(xy.x, xy.y, wh.x, wh.y)
        raise

    def __floordiv__(self, other):
        return self.__truediv__(other)

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            xy = self.xy // other
            wh = self.wh // other
            return BoundingBox(xy.x, xy.y, wh.x, wh.y)
        else:
            raise

    def get_xy2(self):
        return self.xy + self.wh

    def fit_image(self, imgwh):
        self.xy.x = max(0, self.xy.x)
        self.xy.y = max(0, self.xy.y)

        self.wh.x = max(10, min(self.wh.x, imgwh.x - 10))
        self.wh.y = max(10, min(self.wh.y, imgwh.y - 10))
        self.wh.x = min(self.wh.x, imgwh.x - self.xy.x)
        self.wh.y = min(self.wh.y, imgwh.y - self.xy.y)

    def draw(self, img, color=(255, 255, 255)):
        """
        draw bounding box on image
        """
        cv2.rectangle(img, tuple(self.xy), tuple(self.get_xy2()), color, 1)

    def iou(self, other):
        # reference : https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        # determine the (x, y)-coordinates of the intersection rectangle
        if isinstance(other, BoundingBox):
            other_x = other.xy.x
            other_y = other.xy.y
            other_w = other.wh.x
            other_h = other.wh.y
        elif isinstance(other, tuple) or isinstance(other, list) or isinstance(other, np.ndarray):
            other_x, other_y, other_w, other_h = other[:4]
        else:
            raise

        xA = max(self.xy.x, other_x)
        yA = max(self.xy.y, other_y)
        xB = min(self.xy.x + self.wh.x, other_x + other_w)
        yB = min(self.xy.y + self.wh.y, other_y + other_h)

        if xA >= xB or yA >= yB:
            return 0.0

        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = self.wh.x * self.wh.y
        boxBArea = other_w * other_h

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def do_action(self, imgwh, action_idx):
        action_ratios = tuple([ADNetConf.get()['action_move'][x] for x in 'xywh'])

        if action_idx < 8:
            deltas_xy = self.wh * action_ratios[:2]
            deltas_xy.max(1)
            actual_deltas = ADNetwork.ACTIONS[action_idx][:2] * (deltas_xy.x, deltas_xy.y)
            moved_xy = self.xy + actual_deltas
            new_box = BoundingBox(moved_xy.x, moved_xy.y, self.wh.x, self.wh.y)
        elif action_idx == 8:
            new_box = BoundingBox(self.xy.x, self.xy.y, self.wh.x, self.wh.y)
        else:
            deltas_wh = self.wh * action_ratios[2:]
            deltas_wh.max(2)
            deltas_wh_scaled = ADNetwork.ACTIONS[action_idx][2:] * (deltas_wh.x, deltas_wh.y)
            moved_xy = self.xy + -1 * deltas_wh_scaled / 2
            moved_wh = self.wh + deltas_wh_scaled

            new_box = BoundingBox(moved_xy.x, moved_xy.y, moved_wh.x, moved_wh.y)

        if imgwh:
            new_box.fit_image(imgwh)
        return new_box

    def gen_noise_samples(self, imgwh, noise_type, num, **kwargs):
        center_xy = self.xy + self.wh * 0.5
        mean_wh = sum(self.wh) / 2.0

        gaussian_translation_f = kwargs.get('gaussian_translation_f', 0.1)
        uniform_translation_f = kwargs.get('uniform_translation_f', 1)
        uniform_scale_f = kwargs.get('uniform_scale_f', 10)

        samples = []
        if noise_type == 'whole':
            grid_x = range(self.wh.x // 2, imgwh.x - self.wh.x // 2, self.wh.x // 5)
            grid_y = range(self.wh.y // 2, imgwh.y - self.wh.y // 2, self.wh.y // 5)
            samples_tmp = []
            for dx, dy, ds in itertools.product(grid_x, grid_y, range(-5, 5, 1)):
                box = BoundingBox(dx, dy, self.wh.x*(1.05**ds), self.wh.y*(1.05**ds))
                box.fit_image(imgwh)
                samples_tmp.append(box)

            for _ in range(num):
                samples.append(random.choice(samples_tmp))
        else:
            for _ in range(num):
                if noise_type == 'gaussian':
                    dx = gaussian_translation_f * mean_wh * minmax(0.5 * random.normalvariate(0, 1), -1, 1)
                    dy = gaussian_translation_f * mean_wh * minmax(0.5 * random.normalvariate(0, 1), -1, 1)
                    dwh = 1.05 ** (3 * minmax(0.5 * random.normalvariate(0, 1), -1, 1))
                elif noise_type == 'uniform':
                    dx = uniform_translation_f * mean_wh * random.uniform(-1.0, 1.0)
                    dy = uniform_translation_f * mean_wh * random.uniform(-1.0, 1.0)
                    dwh = 1.05 ** (uniform_scale_f * random.uniform(-1.0, 1.0))
                else:
                    raise
                new_cxy = center_xy + (dx, dy)
                new_wh = self.wh * dwh
                box = BoundingBox(new_cxy.x - new_wh.x / 2.0, new_cxy.y - new_wh.y / 2.0, new_wh.x, new_wh.y)
                box.fit_image(imgwh)
                samples.append(box)

        return samples

    def get_posneg_samples(self, imgwh, pos_size, neg_size, use_whole=True, **kwargs):
        pos_thresh = kwargs.get('pos_thresh', ADNetConf.g()['initial_finetune']['pos_thresh'])
        neg_thresh = kwargs.get('neg_thresh', ADNetConf.g()['initial_finetune']['neg_thresh'])

        gaussian_samples = self.gen_noise_samples(imgwh, 'gaussian', pos_size * 2, kwargs=kwargs)
        gaussian_samples = [x for x in gaussian_samples if x.iou(self) > pos_thresh]

        uniform_samples = self.gen_noise_samples(imgwh, 'uniform', neg_size if use_whole else neg_size*2, kwargs=kwargs)
        uniform_samples = [x for x in uniform_samples if x.iou(self) < neg_thresh]

        if use_whole:
            whole_samples = self.gen_noise_samples(imgwh, 'whole', neg_size, kwargs=kwargs)
            whole_samples = [x for x in whole_samples if x.iou(self) < neg_thresh]
        else:
            whole_samples = []

        pos_samples = []
        for _ in range(pos_size):
            pos_samples.append(random.choice(gaussian_samples))

        neg_candidates = uniform_samples + whole_samples
        neg_samples = []
        for _ in range(neg_size):
            neg_samples.append(random.choice(neg_candidates))
        return pos_samples, neg_samples


if __name__ == '__main__':
    ADNetConf.get('./conf/large.yaml')

    # iou test
    box_a = BoundingBox(0, 0, 100, 100)
    box_b = BoundingBox(0, 0, 50, 10)
    assert box_a.iou(box_b) == 0.05

    box_a = BoundingBox(0, 0, 10, 10)
    box_b = BoundingBox(5, 7, 7, 10)
    assert 0.096 < box_a.iou(box_b) < 0.097

    # random generator test
    gt_box = BoundingBox.read_vid_gt('./data/freeman1/')[0]
    gt_box.wh.x = gt_box.wh.y = 30

    imgpath = os.path.join('./data/freeman1/', 'img', '0001.jpg')
    img = cv2.imread(imgpath)

    if False:
        for random_type in ['gaussian', 'uniform', 'whole']:
            gaussian_boxes = gt_box.gen_noise_samples(Coordinate.get_imgwh(img), random_type, 20)

            gt_box.draw(img, BoundingBox.COLOR_GT)
            for box in gaussian_boxes:
                box.draw(img, BoundingBox.COLOR_PREDICT)

            cv2.imshow(random_type, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # pos-neg sample test
    pos, neg = gt_box.get_posneg_samples(Coordinate.get_imgwh(img), 1, 10)
    img = cv2.imread(imgpath)
    for box in pos:
        box.draw(img, BoundingBox.COLOR_PREDICT)
    # for box in neg:
    #     box.draw(img, BoundingBox.COLOR_NEGATIVE)
    gt_box.draw(img, BoundingBox.COLOR_GT)
    actions = BoundingBox.get_action_labels(pos, gt_box)
    cv2.imshow('posneg samples', img)
    cv2.waitKey(10)
    cv2.destroyAllWindows()
