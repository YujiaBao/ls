import torch
import torch.nn as nn
import torch.nn.functional as F

import ls
from ls.utils.print import print

from ls.models.build import ModelFactory


# Register model custom into ls
@ModelFactory.register('custom')
class custom(nn.Module):
    def __init__(self, include_label: bool,
                 num_classes: int,
                 hidden_dim: int,
                 ):

        super().__init__()
        #
        # Note that the arguments include_label and num_classes are required.
        #
        # Num of output classes
        self.num_classes = num_classes
        #
        # If include_label is zero (this is the predictor), we don't append
        # anything to the input.
        # If include_label is non-zero (this is the splitter), we append an
        # 'include_label' dimensional one-hot vector to the input.
        self.include_label = include_label

        # The hidden dim of the input features
        self.hidden_dim = hidden_dim

        if self.include_label > 0:
            # We will append the input feature with the one-hot label embedding.
            self.hidden_dim += self.include_label

        # Construct the modeuls
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x, y=None):
        '''
            For the predictor, the model only receives x during the forward
            pass.
            For the splitter, the model takes both x and y during the forward
            pass.

            Input:
                x: batch_size * hidden_xize
                y: batch_size

            Return:
                x: batch_size * num_classes
        '''

        if self.include_label:
            one_hot = F.one_hot(y, num_classes = self.include_label).float()
            x = torch.cat([x, one_hot], dim=1)

        x = self.classifier(x)

        return x


if __name__ == '__main__':

    data = ls.datasets.Tox21()

    train_data, test_data = ls.learning_to_split(
        data, device='cuda:0', metric='roc_auc', num_classes=2,
        model={
            'name': 'custom',
            'args': {
                'hidden_dim': 1644,
            }
        },
        opt={
            'name': 'Adam',
            'args': {
                'lr': 0.001,
                'weight_decay': 0
            }
        }
    )

