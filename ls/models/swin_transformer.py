import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .build import ModelFactory


class _swin_transformer_base(nn.Module):
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
        self.swin = getattr(models, model_name)(weights=weights)

        # Get the hidden dim
        hidden = self.swin.head.in_features

        # Remove the last layer
        self.swin.head = nn.Identity()

        # Redefine the last classifier
        self.fc = nn.Linear(hidden + self.include_label, self.num_classes)

    def forward(self, x, y=None):

        # Note that swin requires a min size of 224 x 224
        # Users need to resize the input correspondingly.

        x = self.swin(x)

        if self.include_label > 0:
            # Append the one hot label embedding
            x = torch.cat(
                [x, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )

        x = self.fc(x)

        return x


@ModelFactory.register('swin_t')
class swin_t(_swin_transformer_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='swin_t',
                         weight_name='Swin_T_Weights')


@ModelFactory.register('swin_s')
class swin_s(_swin_transformer_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='swin_s',
                         weight_name='Swin_S_Weights')


@ModelFactory.register('swin_b')
class swin_b(_swin_transformer_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='swin_b',
                         weight_name='Swin_B_Weights')
