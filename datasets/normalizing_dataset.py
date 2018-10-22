from torch.utils.data import Dataset


class NormalizingDataset(Dataset):

    def __init__(self, dataset: Dataset, mean, stddev):
        self.dataset = dataset
        self.mean = mean
        self.stddev = stddev

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = x / 255.
        y = y / 255.
        x[:, :, 0] -= self.mean[0]
        x[:, :, 1] -= self.mean[1]
        x[:, :, 2] -= self.mean[2]
        x[:, :, 0] /= self.stddev[0]
        x[:, :, 1] /= self.stddev[1]
        x[:, :, 2] /= self.stddev[2]
        return x, y

    def __len__(self):
        return len(self.dataset)
