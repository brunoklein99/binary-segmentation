import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from data_load import get_filenames_masknames
from datasets.carvana_dataset import CarvanaDataset
from datasets.channel_last_dataset import ChannelLastDataset
from datasets.device_dataset import DeviceDataset
from datasets.normalizing_dataset import NormalizingDataset
from datasets.resize_dataset import ResizeDataset
from datasets.shave_dataset import ShaveDataset
from datasets.tensor_dataset import TensorDataset
from model.unet import UNet


def get_dataset(device):
    filenames, masknames = get_filenames_masknames()

    dataset = CarvanaDataset(
        filenames=filenames,
        masknames=masknames
    )
    dataset = NormalizingDataset(dataset,
                                 mean=[
                                     0.68395978,
                                     0.69088229,
                                     0.69822324
                                 ],
                                 stddev=[
                                     np.square(0.69822324),
                                     np.square(0.0616521),
                                     np.square(0.05961585)
                                 ])
    dataset = ResizeDataset(dataset, size=250)
    dataset = ShaveDataset(dataset, shave_size=5)
    dataset = TensorDataset(dataset)
    dataset = ChannelLastDataset(dataset)
    dataset = DeviceDataset(dataset, device)

    return dataset


def train(params):
    device = torch.device(params['device'])

    dataset = get_dataset(device)

    loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

    model = UNet(n_channels=3, n_classes=1).to(device)

    optimizer = SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9)

    criterion = nn.BCELoss()

    epochs = params['epochs']
    for epoch in range(epochs):
        for i, (x, y) in enumerate(loader):
            y_pred = model(x)

            flat_y = y.view(-1)
            flat_y_pred = y_pred.view(-1)

            loss = criterion(flat_y_pred, flat_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss)


if __name__ == '__main__':
    train({
        'device': 'cuda:0',
        'batch_size': 4,
        'learning_rate': 1e-2,
        'epochs': 10
    })
