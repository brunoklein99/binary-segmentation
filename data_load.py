from os import listdir
from os.path import join


def get_filenames_masknames():
    filenames = [join('data/train', x) for x in listdir('data/train')]
    masknames = [join('data/train_mask', x) for x in listdir('data/train_mask')]
    return filenames, masknames
