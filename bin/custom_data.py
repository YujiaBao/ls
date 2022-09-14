import torch
from torch.utils.data import Dataset

import ls


class CustomData(Dataset):
    def __init__(self):
        # Create a random 1644-dimensional data
        self.data = torch.rand([100, 1644])

        # Create a binary label tensor
        self.targets = torch.cat([
            torch.ones([50]), torch.zeros([50])
        ]).long()

        self.length = len(self.targets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


if __name__ == '__main__':

    data = CustomData()

    train_data, test_data = ls.learning_to_split(
        data, device='cuda:0', num_classes=2, model={'name': 'mlp'})

