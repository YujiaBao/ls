import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import ModelFactory


@ModelFactory.register('mlp')
class mlp(nn.Module):
    '''
        A simple multi layer perceptron with dropout
    '''
    def __init__(self, include_label: int,
                 hidden_dim_list: list[int] = [1644, 1024, 1024, 1024],
                 num_classes: int = 2,
                 dropout: float = 0.3):

        super().__init__()

        # List of the hidden dimension sizes
        self.hidden_dim_list = hidden_dim_list.copy()

        # If zero, we don't append anything to the input. If non-zero, we append
        # an include_label dimensional one-hot vector to the input.
        self.include_label = include_label
        if self.include_label > 0:
            # We will append the input feature with the one-hot label embedding.
            self.hidden_dim_list[0] += self.include_label

        # Dropout rate
        self.dropout = dropout

        # Construct the modeuls
        modules = []
        for i in range(len(self.hidden_dim_list)):
            if i != 0:
                modules.append(nn.Dropout(self.dropout))

            if i != len(self.hidden_dim_list) - 1:
                modules.append(nn.Linear(self.hidden_dim_list[i],
                                         self.hidden_dim_list[i+1]))
                modules.append(nn.ReLU())
            else:
                # Final layer
                modules.append(nn.Linear(self.hidden_dim_list[i], num_classes))

        self.seq = nn.Sequential(*modules)

    def forward(self, x, y=None):
        '''
            Input:
                x: batch_size * hidden_xize
                y: batch_size

            Return:
                x: batch_size * num_classes
        '''
        if self.include_label:
            x = torch.cat(
                [x, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )

        x = self.seq(x)

        return x

