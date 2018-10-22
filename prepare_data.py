from os.path import join, splitext

import cv2
import numpy as np


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def str_encoding_to_tuple_list(enc):
    enc = enc.split()
    enc = chunks(enc, n=2)
    for p, c in enc:
        yield int(p), int(c)


def get_mask(width, height, str_encoding):
    e = str_encoding_to_tuple_list(str_encoding)
    m = np.zeros(shape=(width * height))
    for p, c in e:
        m[p:p + c] = 255
    m = np.reshape(m, newshape=(height, width))
    return m


def get_mask_filename(img_filename):
    name, ext = splitext(img_filename)
    return '{}_mask{}'.format(name, ext)


if __name__ == '__main__':
    width = 1918
    height = 1280
    with open('data/train_masks.csv') as f:
        f.readline()
        count = 0
        for line in f:
            filename, encoding = line.split(sep=',')
            filename = join('data/train_mask', filename)
            mask = get_mask(width=width, height=height, str_encoding=encoding)
            cv2.imwrite(filename, mask)
            count += 1
            print('saving mask {}'.format(count))