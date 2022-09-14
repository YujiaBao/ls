import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .build import ModelFactory


@ModelFactory.register('mobilenet_v2')
class mobilenet_v2(nn.Module):
    '''
        Add splitter / predictor support for the mnasnet in torchvision.models
    '''
    def __init__(self,
                 include_label: int,
                 num_classes: int,
                 model_name: str = 'mobilenet_v2',
                 weight_name: str = 'MobileNet_V2_Weights'):
        super().__init__()

        self.include_label = include_label
        self.num_classes = num_classes

        # Get the name of the weights
        weights = getattr(models, weight_name).DEFAULT

        # Load the pre-trained model
        self.mobilenet = getattr(models, model_name)(weights=weights)

        # Remove the last layer
        self.mobilenet.classifier[-1] = nn.Identity()

        hidden = self.mobilenet.last_channel

        # Redefine the last classifier
        self.fc = nn.Linear(hidden + self.include_label, self.num_classes)

    def forward(self, x, y=None):

        x = self.mobilenet(x)

        if self.include_label > 0:
            # Append the one hot label embedding
            x = torch.cat(
                [x, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )

        x = self.fc(x)

        return x

