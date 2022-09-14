import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .build import ModelFactory


@ModelFactory.register('inception_v3')
class inception_v3(nn.Module):
    '''
        Add splitter / predictor support for the googlenet in torchvision.models
    '''
    def __init__(self,
                 include_label: int,
                 num_classes: int,
                 model_name: str = 'inception_v3',
                 weight_name: str = 'Inception_V3_Weights'):
        super().__init__()

        self.include_label = include_label
        self.num_classes = num_classes

        # Get the name of the weights
        weights = getattr(models, weight_name).DEFAULT

        # Load the googlenet
        self.inception = getattr(models, model_name)(weights=weights)

        # Turn off AuxLogits so that we can work with smaller datasets
        self.inception.AuxLogits = None

        # Skip the last fc layer
        self.inception.fc = nn.Identity()

        # We need to concatenate the last layer with the one-hot label embedding
        # (for the splitter)
        hidden = 2048  # inception last layer has 2048 hidden dim
        self.fc = nn.Linear(hidden + self.include_label, self.num_classes)

    def forward(self, x, y=None):

        x = self.inception(x)

        if self.training:
            # inception will return an inception_output (named tuple) if it is
            # in training mode
            x = x.logits

        if self.include_label > 0:
            # Append the one hot label embedding
            x = torch.cat(
                [x, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )

        x = self.fc(x)

        return x
