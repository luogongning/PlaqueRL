import random

import cv2
import numpy as np

from configs import ADNetConf


def imread(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = BGR2RGB(img)
    return img


def BGR2RGB(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def extract_region(img, bbox):
    xy_center = bbox.xy + bbox.wh * 0.5

    wh = bbox.wh * ADNetConf.get()['predict']['roi_zoom']
    xy = xy_center - wh * 0.5
    xy.x = max(xy.x, 0)
    xy.y = max(xy.y, 0)

    # crop and resize
    crop = img[xy.y:xy.y+wh.y, xy.x:xy.x+wh.x, :]
    resize = cv2.resize(crop, (112, 112))
    return resize


def minmax(num, min_num, max_num):
    return max(min(num, max_num), min_num)


def choices(seq, l):
    # for support python2
    return [random.choice(seq) for _ in range(l)]


def random_idxs(max, k):
    if k >= max:
        return [random.randint(0, max - 1) for _ in range(k)]
    else:
        l = list(range(max))
        random.shuffle(l)
        return l[:k]


def choices_by_idx(seq, idxs):
    return [seq[x] for x in idxs]


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def onehot(idxs):
    a = np.zeros(shape=(1, len(idxs), 11), dtype=np.int8)
    # a[0, np.arange(len(idxs)), idxs] = 1
    for i, idx in enumerate(idxs):
        if idx >= 12 or idx < 0:
            continue
        a[0, i, idx] = 1
    return a


def onehot_flatten(idxs):
    a = onehot(idxs)
    a = a.reshape((1, 1, a.shape[1]*a.shape[2]))
    return a


def imshow_grid(title, images, cols, rows):
    h, w = images[0].shape[:2]
    canvas = np.zeros((rows*h, cols*w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        y = (i // cols) * h
        x = (i % cols) * w
        canvas[y:y+h, x:x+w] = img
    cv2.imshow(title, canvas)
