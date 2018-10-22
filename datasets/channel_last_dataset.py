from torch.utils.data import Dataset


class ChannelLastDataset(Dataset):

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = x.permute(2, 1, 0)
        y = y.permute(2, 1, 0)
        return x, y

    def __len__(self):
        return len(self.dataset)
