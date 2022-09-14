import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .build import ModelFactory


class _mnasnet_base(nn.Module):
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
        self.mnasnet = getattr(models, model_name)(weights=weights)

        # Remove the last layer
        self.mnasnet.classifier[-1] = nn.Identity()

        hidden = 1280

        # Redefine the last classifier
        self.fc = nn.Linear(hidden + self.include_label, self.num_classes)

    def forward(self, x, y=None):

        x = self.mnasnet(x)

        if self.include_label > 0:
            # Append the one hot label embedding
            x = torch.cat(
                [x, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )

        x = self.fc(x)

        return x


@ModelFactory.register('mnasnet0_5')
class mnasnet0_5(_mnasnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='mnasnet0_5',
                         weight_name='MNASNet0_5_Weights')


@ModelFactory.register('mnasnet0_75')
class mnasnet0_75(_mnasnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='mnasnet0_75',
                         weight_name='MNASNet0_75_Weights')


@ModelFactory.register('mnasnet1_0')
class mnasnet1_0(_mnasnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='mnasnet1_0',
                         weight_name='MNASNet1_0_Weights')


@ModelFactory.register('mnasnet1_3')
class mnasnet1_3(_mnasnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='mnasnet1_3',
                         weight_name='MNASNet1_3_Weights')

