import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .build import ModelFactory


class _vgg_base(nn.Module):
    '''
        Add splitter / predictor support for the vgg in torchvision.models
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
        self.vgg = getattr(models, model_name)(weights=weights)

        # Remove the last layer
        self.vgg.classifier[-1] = nn.Identity()

        hidden = 4096  # vgg uses a hidden dim of 4096

        # Redefine the last classifier
        self.fc = nn.Linear(hidden + self.include_label, self.num_classes)

    def forward(self, x, y=None):

        x = self.vgg(x)

        if self.include_label > 0:
            # Append the one hot label embedding
            x = torch.cat(
                [x, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )

        x = self.fc(x)

        return x


@ModelFactory.register('vgg11')
class vgg11(_vgg_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='vgg11',
                         weight_name='VGG11_Weights')


@ModelFactory.register('vgg11_bn')
class vgg11_bn(_vgg_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='vgg11_bn',
                         weight_name='VGG11_BN_Weights')


@ModelFactory.register('vgg13')
class vgg13(_vgg_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='vgg13',
                         weight_name='VGG13_Weights')


@ModelFactory.register('vgg13_bn')
class vgg13_bn(_vgg_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='vgg13_bn',
                         weight_name='VGG13_BN_Weights')

@ModelFactory.register('vgg16')
class vgg16(_vgg_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='vgg16',
                         weight_name='VGG16_Weights')


@ModelFactory.register('vgg16_bn')
class vgg16_bn(_vgg_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='vgg16_bn',
                         weight_name='VGG16_BN_Weights')


@ModelFactory.register('vgg19')
class vgg19(_vgg_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='vgg19',
                         weight_name='VGG19_Weights')


@ModelFactory.register('vgg19_bn')
class vgg19_bn(_vgg_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='vgg19_bn',
                         weight_name='VGG19_BN_Weights')

