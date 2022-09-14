import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .build import ModelFactory


class _vision_transformer_base(nn.Module):
    '''
        Add splitter / predictor support for the vit in torchvision.models
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
        self.vit = getattr(models, model_name)(weights=weights)

        # Get the last hidden dim size
        hidden = self.vit.heads[-1].in_features

        # Remove the last layer
        self.vit.heads[-1] = nn.Identity()

        # Redefine the last classifier
        self.fc = nn.Linear(hidden + self.include_label, self.num_classes)

    def forward(self, x, y=None):

        # Note that ViT requires the input to be batch_size x 3 x 224 x 224
        # Users need to resize the input correspondingly.
        x = self.vit(x)

        if self.include_label > 0:
            # Append the one hot label embedding
            x = torch.cat(
                [x, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )

        x = self.fc(x)

        return x


@ModelFactory.register('vit_b_16')
class vit_b_16(_vision_transformer_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='vit_b_16',
                         weight_name='ViT_B_16_Weights')


@ModelFactory.register('vit_b_32')
class vit_b_32(_vision_transformer_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='vit_b_32',
                         weight_name='ViT_B_32_Weights')


@ModelFactory.register('vit_l_16')
class vit_l_16(_vision_transformer_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='vit_l_16',
                         weight_name='ViT_L_16_Weights')


@ModelFactory.register('vit_l_32')
class vit_l_32(_vision_transformer_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='vit_l_32',
                         weight_name='ViT_L_32_Weights')


@ModelFactory.register('vit_h_14')
class vit_h_14(_vision_transformer_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='vit_h_14',
                         weight_name='ViT_H_14_Weights')
