import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .build import ModelFactory


class _mobilenet_v3_base(nn.Module):
    '''
        Add splitter / predictor support for the mnasnet in torchvision.models
    '''
    def __init__(self,
                 include_label: int,
                 num_classes: int,
                 model_name: str,
                 weight_name: str):
        super().__init__()

        self.include_label = include_label
        self.num_classes = num_classes

        # Get the name of the weights
        weights = getattr(models, weight_name).DEFAULT

        # Load the pre-trained model
        self.mobilenet = getattr(models, model_name)(weights=weights)

        # Remove the last layer
        hidden = self.mobilenet.classifier[-1].in_features
        self.mobilenet.classifier[-1] = nn.Identity()

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


@ModelFactory.register('mobilenet_v3_large')
class mobilenet_v3_large(_mobilenet_v3_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='mobilenet_v3_large',
                         weight_name='MobileNet_V3_Large_Weights')


@ModelFactory.register('mobilenet_v3_small')
class mobilenet_v3_small(_mobilenet_v3_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='mobilenet_v3_small',
                         weight_name='MobileNet_V3_Small_Weights')

