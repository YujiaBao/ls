import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .build import ModelFactory


@ModelFactory.register('googlenet')
class googlenet(nn.Module):
    '''
        Add splitter / predictor support for the googlenet in torchvision.models
    '''
    def __init__(self,
                 include_label: int,
                 num_classes: int,
                 model_name: str = 'googlenet',
                 weight_name: str = 'GoogLeNet_Weights'):
        super().__init__()

        self.include_label = include_label
        self.num_classes = num_classes

        # Get the name of the weights
        weights = getattr(models, weight_name).DEFAULT

        # Load the googlenet
        self.googlenet = getattr(models, model_name)(weights=weights)

        # Skip the last fc layer
        self.googlenet.fc = nn.Identity()

        # We need to concatenate the last layer with the one-hot label embedding
        # (for the splitter)
        hidden = 1024  # googlenet last layer has 1024 hidden dim
        self.fc = nn.Linear(hidden + self.include_label, self.num_classes)

    def forward(self, x, y=None):

        x = self.googlenet(x)

        if self.include_label > 0:
            # Append the one hot label embedding
            x = torch.cat(
                [x, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )

        x = self.fc(x)

        return x
