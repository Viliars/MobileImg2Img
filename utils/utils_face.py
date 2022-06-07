# -*- coding: utf-8 -*-
import numpy as np
import cv2
import random


def random_crop(lq, hq, patch_size=140):
    h, w = lq.shape[:2]
    rnd_h = random.randint(0, h-patch_size)
    rnd_w = random.randint(0, w-patch_size)

    lq = lq[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
    hq = hq[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]

    return lq, hq


def get_face_pair(img_LQ, img_HQ, patch_size=140):

    scale = round(random.uniform(1.0, 4.0), 2)

    size = int(1024//round(random.uniform(1.0, 4.0), 2))
    interpolation = random.choice([1, 2, 3])

    # resize
    img_LQ = cv2.resize(img_LQ, (size, size), interpolation=interpolation)
    img_HQ = cv2.resize(img_HQ, (size, size), interpolation=interpolation)

    # random crop
    img_LQ, img_HQ = random_crop(img_LQ, img_HQ, patch_size)

    return img_LQ, img_HQ



if __name__ == '__main__':
    from utils import utils_image as util
    hq = util.imread_uint('~/Documents/FFHQ/Train/HQ/10000.png', 3)
    lq = util.imread_uint('~/Documents/FFHQ/Train/LQ/10000.png', 3)
    hq = util.uint2single(hq)
    lq = util.uint2single(lq)


    img_lq, img_hq = get_face_pair(lq, hq, 140)
    img_lq = util.single2uint(img_lq)
    img_hq = util.single2uint(img_hq)

    img_concat = np.concatenate([img_lq, img_hq], axis=1)
    util.imsave(img_concat, 'test.png')
