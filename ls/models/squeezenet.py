import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .build import ModelFactory


class _squeezenet_base(nn.Module):
    '''
        Add splitter / predictor support for the squeezenet in torchvision.models
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
        self.squeezenet = getattr(models, model_name)(weights=weights)

        # Redefine the last classifier
        final_conv = nn.Conv2d(512+self.include_label, self.num_classes,
                               kernel_size=1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.squeezenet.classifier[0].p),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)

    def forward(self, x, y=None):

        x = self.squeezenet.features(x)

        if self.include_label > 0:
            # Append the one hot label embedding
            one_hot = F.one_hot(y, num_classes=self.include_label).float()

            # Unsqueeze so that the dim is batch x num_classes x 1 x 1
            one_hot = one_hot.unsqueeze(-1).unsqueeze(-1)

            # Expand the one_hot embedding so that it aligns with the features
            _, _, h, w = x.size()
            x = torch.cat([x, one_hot.expand(-1, -1, h, w)], dim=1)

        x = self.classifier(x)
        x = torch.flatten(x, 1)

        return x


@ModelFactory.register('squeezenet1_0')
class squeezenet1_0(_squeezenet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='squeezenet1_0',
                         weight_name='SqueezeNet1_0_Weights')


@ModelFactory.register('squeezenet1_1')
class squeezenet1_1(_squeezenet_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, model_name='squeezenet1_1',
                         weight_name='SqueezeNet1_1_Weights')
