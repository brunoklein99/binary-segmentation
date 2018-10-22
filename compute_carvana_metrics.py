import cv2
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from data_load import get_filenames_masknames
from datasets.carvana_dataset import CarvanaDataset

if __name__ == '__main__':
    filenames, masknames = get_filenames_masknames()

    dataset = CarvanaDataset(
        filenames=filenames,
        masknames=masknames
    )

    r_scaler = StandardScaler()
    g_scaler = StandardScaler()
    b_scaler = StandardScaler()

    for i in tqdm(range(len(dataset))):
        img, _ = dataset[i]
        img = img / 255
        r = np.reshape(img[:, :, 0], newshape=(-1, 1))
        g = np.reshape(img[:, :, 1], newshape=(-1, 1))
        b = np.reshape(img[:, :, 2], newshape=(-1, 1))
        r_scaler.partial_fit(r)
        g_scaler.partial_fit(g)
        b_scaler.partial_fit(b)

    print('r mean: ', r_scaler.mean_)
    print('r var:  ', r_scaler.var_)

    print('g mean: ', g_scaler.mean_)
    print('g var:  ', g_scaler.var_)

    print('b mean: ', b_scaler.mean_)
    print('b var:  ', b_scaler.var_)
