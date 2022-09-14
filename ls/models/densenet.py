import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .build import ModelFactory


class _densenet_base(nn.Module):
    '''
        Add splitter / predictor support for the densenet in torchvision.models
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
        self.densenet = getattr(models, model_name)(weights=weights)

        # Get the last layer's dimension
        hidden = self.densenet.classifier.in_features

        # Redefine the last classifier
        self.fc = nn.Linear(hidden + self.include_label, self.num_classes)

    def forward(self, x, y=None):
        '''
            https://pytorch.org/vision/stable/_modules/torchvision/models/densenet.html#densenet121
        '''
        x = self.densenet.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptifeaturesve_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        if self.include_label > 0:
            # Append the one hot label embedding
            x = torch.cat(
                [x, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )

        x = self.fc(x)

        return x


@ModelFactory.register('densenet121')
class densenet121(_densenet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='densenet121',
                         weight_name='DenseNet121_Weights')


@ModelFactory.register('densenet161')
class densenet161(_densenet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='densenet161',
                         weight_name='DenseNet161_Weights')


@ModelFactory.register('densenet169')
class densenet169(_densenet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='densenet169',
                         weight_name='DenseNet169_Weights')


@ModelFactory.register('densenet201')
class densenet201(_densenet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='densenet201',
                         weight_name='DenseNet201_Weights')

