import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import Vocab

from .build import ModelFactory


@ModelFactory.register('textcnn')
class textcnn(nn.Module):
    '''
        1D CNN followed by max-pooling-over-time.
        https://arxiv.org/abs/1408.5882
    '''
    def __init__(self,
                 include_label: int,
                 num_classes: int,
                 embedding_dim: int = 300,
                 num_filters: int = 50,
                 filter_sizes: list[int] = [3, 4, 5],
                 dropout=0.1):

        super().__init__()

        self.include_label = include_label
        self.num_classes = num_classes

        # Init the conv filters
        # We use num_filters for each filter size
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters,
                      kernel_size=k)
            for k in filter_sizes
        ])

        # get the final mlp layer
        hidden_dim = num_filters * len(filter_sizes)
        self.seq = nn.Sequential(
            nn.Linear(hidden_dim + self.include_label, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, y=None):
        '''
            @param text: batch_size * max_text_len
            @return output: batch_size * embedding_dim
        '''
        # Apply all filters
        # [batch_size, num_filters, seq_len] * len(filter_sizes)
        x = [conv(x) for conv in self.convs]

        # Max pool over time
        # batch_size, hidden_dim
        x = torch.cat([ torch.max(feat, dim=2)[0] for feat in x ], dim=1)

        # Final classifier
        if self.include_label:
            x = torch.cat(
                [x, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )

        x = self.seq(x)

        return x


