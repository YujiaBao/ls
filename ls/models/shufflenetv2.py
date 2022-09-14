import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .build import ModelFactory


class _shufflenetv2_base(nn.Module):
    '''
        Add splitter / predictor support for the shufflenetv2 in
        torchvision.models
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
        self.shufflenetv2 = getattr(models, model_name)(weights=weights)

        # Get the hidden size
        hidden = self.shufflenetv2.fc.in_features

        # Remove the last layer
        self.shufflenetv2.fc = nn.Identity()

        # Redefine the last classifier
        self.fc = nn.Linear(hidden + self.include_label, self.num_classes)

    def forward(self, x, y=None):

        x = self.shufflenetv2(x)

        if self.include_label > 0:
            # Append the one hot label embedding
            x = torch.cat(
                [x, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )

        x = self.fc(x)

        return x


@ModelFactory.register('shufflenet_v2_x0_5')
class shufflenet_v2_x0_5(_shufflenetv2_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='shufflenet_v2_x0_5',
                         weight_name='ShuffleNet_V2_X0_5_Weights')


@ModelFactory.register('shufflenet_v2_x1_0')
class shufflenet_v2_x1_0(_shufflenetv2_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='shufflenet_v2_x1_0',
                         weight_name='ShuffleNet_V2_X1_0_Weights')


@ModelFactory.register('shufflenet_v2_x1_5')
class shufflenet_v2_x1_5(_shufflenetv2_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='shufflenet_v2_x1_5',
                         weight_name='ShuffleNet_V2_X1_5_Weights')


@ModelFactory.register('shufflenet_v2_x2_0')
class shufflenet_v2_x2_0(_shufflenetv2_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='shufflenet_v2_x2_0',
                         weight_name='ShuffleNet_V2_X2_0_Weights')
