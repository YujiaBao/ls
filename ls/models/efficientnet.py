import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .build import ModelFactory


class _efficientnet_base(nn.Module):
    '''
        Add splitter / predictor support for the efficientnets in torchvision.models
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
        # For efficientnet_v2_s, it will be
        # weights = EfficientNet_V2_S_Weights.DEFAULT
        weights = getattr(models, weight_name).DEFAULT

        # Load the efficient net
        efficientnet = getattr(models, model_name)(weights=weights)

        # Get the dropout rate from the efficient net
        dropout_rate = efficientnet.classifier[0].p
        # Get the output dimension from the efficient net
        hidden = efficientnet.classifier[1].in_features

        # Load all children but the last classifier
        self.efficientnet = nn.Sequential(*(
            list(efficientnet.children())[:-1]
            + [nn.Dropout(p=dropout_rate, inplace=True)]
        ))

        # Redefine the last classifier
        self.fc = nn.Linear(hidden + self.include_label, self.num_classes)

    def forward(self, x, y=None):

        x = self.efficientnet(x)

        x = torch.flatten(x, 1)

        if self.include_label > 0:
            # Append the one hot label embedding
            x = torch.cat(
                [x, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )

        x = self.fc(x)

        return x


@ModelFactory.register('efficientnet_v2_s')
class efficientnet_v2_s(_efficientnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='efficientnet_v2_s',
                         weight_name='EfficientNet_V2_S_Weights')


@ModelFactory.register('efficientnet_v2_m')
class efficientnet_v2_m(_efficientnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='efficientnet_v2_m',
                         weight_name='EfficientNet_V2_M_Weights')

@ModelFactory.register('efficientnet_v2_l')
class efficientnet_v2_l(_efficientnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='efficientnet_v2_l',
                         weight_name='EfficientNet_V2_L_Weights')

@ModelFactory.register('efficientnet_b0')
class efficientnet_b0(_efficientnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='efficientnet_b0',
                         weight_name='EfficientNet_B0_Weights')

@ModelFactory.register('efficientnet_b1')
class efficientnet_b1(_efficientnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='efficientnet_b1',
                         weight_name='EfficientNet_B1_Weights')

@ModelFactory.register('efficientnet_b2')
class efficientnet_b2(_efficientnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='efficientnet_b2',
                         weight_name='EfficientNet_B2_Weights')

@ModelFactory.register('efficientnet_b3')
class efficientnet_b3(_efficientnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='efficientnet_b3',
                         weight_name='EfficientNet_B3_Weights')

@ModelFactory.register('efficientnet_b4')
class efficientnet_b4(_efficientnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='efficientnet_b4',
                         weight_name='EfficientNet_B4_Weights')

@ModelFactory.register('efficientnet_b5')
class efficientnet_b5(_efficientnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='efficientnet_b5',
                         weight_name='EfficientNet_B5_Weights')

@ModelFactory.register('efficientnet_b6')
class efficientnet_b6(_efficientnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='efficientnet_b6',
                         weight_name='EfficientNet_B6_Weights')

@ModelFactory.register('efficientnet_b7')
class efficientnet_b7(_efficientnet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='efficientnet_b7',
                         weight_name='EfficientNet_B7_Weights')
