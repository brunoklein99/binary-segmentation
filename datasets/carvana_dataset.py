import cv2

from torch.utils.data import Dataset


class CarvanaDataset(Dataset):

    def __init__(self, filenames, masknames):
        self.filenames = filenames
        self.masknames = masknames
        assert len(filenames) == len(masknames)

    def __getitem__(self, index):
        x = cv2.imread(self.filenames[index])
        y = cv2.imread(self.masknames[index])
        return x, y

    def __len__(self):
        return len(self.filenames)
