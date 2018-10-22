from torch.utils.data import Dataset


class ShaveDataset(Dataset):

    def __init__(self, dataset: Dataset, shave_size):
        self.dataset = dataset
        self.shave_size = shave_size

    def __getitem__(self, index):
        x, y = self.dataset[index]
        s = self.shave_size
        y = y[s:-s, s:-s, :]
        return x, y

    def __len__(self):
        return len(self.dataset)