import cv2
import numpy as np
from torch.utils.data import Dataset


class ResizeDataset(Dataset):

    def __init__(self, dataset: Dataset, size):
        self.dataset = dataset
        self.size = size

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = cv2.resize(x, dsize=(self.size, self.size))
        y = cv2.resize(y, dsize=(self.size, self.size))
        y = np.expand_dims(y[:, :, 0], axis=-1)
        return x, y

    def __len__(self):
        return len(self.dataset)
