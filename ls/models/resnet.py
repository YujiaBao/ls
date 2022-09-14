import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .build import ModelFactory


class _resnet_base(nn.Module):
    '''
        Add splitter / predictor support for the resnets in torchvision.models
    '''
    def __init__(self,
                 include_label: int,
                 num_classes: int,
                 model_name: str,
                 weight_name: str):
        super().__init__()

        self.include_label = include_label
        self.num_classes = num_classes

        # Get the name of the weights for the resnet
        # For resnet50, it will be
        # weights = ResNet50_Weights.DEFAULT
        weights = getattr(models, weight_name).DEFAULT

        # Load the resnet
        resnet = getattr(models, model_name)(weights=weights)

        hidden = resnet.fc.in_features
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1]))
        self.fc = nn.Linear(hidden + self.include_label, self.num_classes)

    def forward(self, x, y=None):

        x = self.resnet(x)

        x = torch.flatten(x, 1)

        if self.include_label > 0:
            # Append the one hot label embedding
            x = torch.cat(
                [x, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )

        x = self.fc(x)

        return x


@ModelFactory.register('resnet18')
class resnet18(_resnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='resnet18',
                         weight_name='ResNet18_Weights')


@ModelFactory.register('resnet34')
class resnet34(_resnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='resnet34',
                         weight_name='ResNet34_Weights')


@ModelFactory.register('resnet50')
class resnet50(_resnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='resnet50',
                         weight_name='ResNet50_Weights')


@ModelFactory.register('resnet101')
class resnet101(_resnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='resnet101',
                         weight_name='ResNet101_Weights')


@ModelFactory.register('resnet152')
class resnet152(_resnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='resnet152',
                         weight_name='ResNet152_Weights')


@ModelFactory.register('resnext50_32x4d')
class resnext50_32x4d(_resnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='resnext50_32x4d',
                         weight_name='ResNeXt50_32X4D_Weights')


@ModelFactory.register('resnext101_32x8d')
class resnext101_32x8d(_resnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='resnext101_32x8d',
                         weight_name='ResNeXt101_32X8D_Weights')


@ModelFactory.register('resnext101_64x4d')
class resnext101_64x4d(_resnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='resnext101_64x4d',
                         weight_name='ResNeXt101_64X4D_Weights')


@ModelFactory.register('wide_resnet50_2')
class wide_resnet50_2(_resnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='wide_resnet50_2',
                         weight_name='Wide_ResNet50_2_Weights')


@ModelFactory.register('wide_resnet101_2')
class wide_resnet101_2(_resnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='wide_resnet101_2',
                         weight_name='Wide_ResNet101_2_Weights')

