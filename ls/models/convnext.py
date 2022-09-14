import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .build import ModelFactory


class _convnext_base(nn.Module):
    '''
        Add splitter / predictor support for the convnext in torchvision.models
    '''
    def __init__(self,
                 include_label: int,
                 num_classes: int,
                 model_name: str,
                 weight_name: str):
        super().__init__()

        self.include_label = include_label
        self.num_classes = num_classes

        # Get the name of the weights for the convnext
        weights = getattr(models, weight_name).DEFAULT

        # Load the pre-trained model
        self.convnext = getattr(models, model_name)(weights=weights)

        # Get the last layer's dimension
        hidden = self.convnext.classifier[-1].in_features

        # Remove the last layer
        self.convnext.classifier = nn.Sequential(*(
            list(self.convnext.classifier.children())[:-1]
        ))

        # Redefine the last classifier
        self.fc = nn.Linear(hidden + self.include_label, self.num_classes)

    def forward(self, x, y=None):

        x = self.convnext(x)

        if self.include_label > 0:
            # Append the one hot label embedding
            x = torch.cat(
                [x, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )

        x = self.fc(x)

        return x


@ModelFactory.register('convnext_tiny')
class convnext_tiny(_convnext_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='convnext_tiny',
                         weight_name='ConvNeXt_Tiny_Weights')


@ModelFactory.register('convnext_small')
class convnext_small(_convnext_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='convnext_small',
                         weight_name='ConvNeXt_Small_Weights')


@ModelFactory.register('convnext_base')
class convnext_base(_convnext_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='convnext_base',
                         weight_name='ConvNeXt_Base_Weights')


@ModelFactory.register('convnext_large')
class convnext_large(_convnext_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='convnext_large',
                         weight_name='ConvNeXt_Large_Weights')

