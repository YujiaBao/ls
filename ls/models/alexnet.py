import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .build import ModelFactory


@ModelFactory.register('alexnet')
class alexnet(nn.Module):
    '''
        Add splitter / predictor support for the alexnet in torchvision.models
    '''
    def __init__(self,
                 include_label: int,
                 num_classes: int,
                 model_name: str = 'alexnet',
                 weight_name: str = 'AlexNet_Weights'):
        super().__init__()

        self.include_label = include_label
        self.num_classes = num_classes

        # Get the name of the weights
        weights = getattr(models, weight_name).DEFAULT

        # Load the alexnet
        self.alexnet = getattr(models, model_name)(weights=weights)

        # For the classifier layer, we take all but the last linear layer
        self.alexnet.classifier = nn.Sequential(
            *list(self.alexnet.classifier.children())[:-1]
        )

        # We need to concatenate the last layer with the one-hot label embedding
        # (for the splitter)
        hidden = 4096  # AlexNet last layer has 4096 hidden dim
        self.fc = nn.Linear(hidden + self.include_label, self.num_classes)

    def forward(self, x, y=None):

        x = self.alexnet(x)

        if self.include_label > 0:
            # Append the one hot label embedding
            x = torch.cat(
                [x, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )

        x = self.fc(x)

        return x
