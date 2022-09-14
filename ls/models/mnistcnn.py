import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import ModelFactory


@ModelFactory.register('mnistcnn')
class mnistcnn(nn.Module):
    '''
        A simple CNN for MNIST
        https://github.com/pytorch/examples/blob/main/mnist/main.py
    '''
    def __init__(self, include_label: int,
                 num_classes: int,
                 hidden_dim: int = 100,
                 input_channels: int = 1):
        super().__init__()

        self.input_channels = input_channels

        self.hidden_dim = hidden_dim

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.include_label = include_label

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.input_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.seq = nn.Sequential(
            nn.Linear(9216 + self.include_label, self.hidden_dim),  # include the label
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def forward(self, x, y=None):
        '''
            Input:
                x: batch_size x 1 x 28 x 28
                y: batch_size
            Return:
                x: batch_size x num_classes
        '''
        x = x.view(x.shape[0], self.input_channels, 28, 28)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        if self.include_label:
            x = torch.cat([x, F.one_hot(y, num_classes=self.include_label).float()], dim=1)

        x = self.seq(x)

        return x
