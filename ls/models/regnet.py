import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .build import ModelFactory


class _regnet_base(nn.Module):
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
        self.regnet = getattr(models, model_name)(weights=weights)

        # Get the hidden dim
        hidden = self.regnet.fc.in_features

        # Remove the last layer
        self.regnet.fc = nn.Identity()

        # Redefine the last classifier
        self.fc = nn.Linear(hidden + self.include_label, self.num_classes)

    def forward(self, x, y=None):

        x = self.regnet(x)

        if self.include_label > 0:
            # Append the one hot label embedding
            x = torch.cat(
                [x, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )

        x = self.fc(x)

        return x


@ModelFactory.register('regnet_y_400mf')
class regnet_y_400mf(_regnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='regnet_y_400mf',
                         weight_name='RegNet_Y_400MF_Weights')


@ModelFactory.register('regnet_y_800mf')
class regnet_y_800mf(_regnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='regnet_y_800mf',
                         weight_name='RegNet_Y_800MF_Weights')


@ModelFactory.register('regnet_y_1_6gf')
class regnet_y_1_6gf(_regnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='regnet_y_1_6gf',
                         weight_name='RegNet_Y_1_6GF_Weights')


@ModelFactory.register('regnet_y_3_2gf')
class regnet_y_3_2gf(_regnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='regnet_y_3_2gf',
                         weight_name='RegNet_Y_3_2GF_Weights')


@ModelFactory.register('regnet_y_8gf')
class regnet_y_8gf(_regnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='regnet_y_8gf',
                         weight_name='RegNet_Y_8GF_Weights')


@ModelFactory.register('regnet_y_16gf')
class regnet_y_16gf(_regnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='regnet_y_16gf',
                         weight_name='RegNet_Y_16GF_Weights')


@ModelFactory.register('regnet_y_32gf')
class regnet_y_32gf(_regnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='regnet_y_32gf',
                         weight_name='RegNet_Y_32GF_Weights')


@ModelFactory.register('regnet_y_128gf')
class regnet_y_128gf(_regnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='regnet_y_128gf',
                         weight_name='RegNet_Y_128GF_Weights')


@ModelFactory.register('regnet_x_400mf')
class regnet_x_400mf(_regnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='regnet_x_400mf',
                         weight_name='RegNet_X_400MF_Weights')


@ModelFactory.register('regnet_x_800mf')
class regnet_x_800mf(_regnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='regnet_x_800mf',
                         weight_name='RegNet_X_800MF_Weights')


@ModelFactory.register('regnet_x_1_6gf')
class regnet_x_1_6gf(_regnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='regnet_x_1_6gf',
                         weight_name='RegNet_X_1_6GF_Weights')


@ModelFactory.register('regnet_x_3_2gf')
class regnet_x_3_2gf(_regnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='regnet_x_3_2gf',
                         weight_name='RegNet_X_3_2GF_Weights')


@ModelFactory.register('regnet_x_8gf')
class regnet_x_8gf(_regnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='regnet_x_8gf',
                         weight_name='RegNet_X_8GF_Weights')


@ModelFactory.register('regnet_x_16gf')
class regnet_x_16gf(_regnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='regnet_x_16gf',
                         weight_name='RegNet_X_16GF_Weights')


@ModelFactory.register('regnet_x_32gf')
class regnet_x_32gf(_regnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='regnet_x_32gf',
                         weight_name='RegNet_X_32GF_Weights')
