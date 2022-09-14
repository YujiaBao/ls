import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from .build import ModelFactory


@ModelFactory.register('bert')
class bert(nn.Module):
    def __init__(self,
                 include_label: int,
                 num_classes: int,
                 pretrained_model_name_or_path: str = 'bert-base-uncased'):

        super().__init__()

        self.include_label = include_label
        self.num_classes = num_classes

        # Load the pre-trained model
        self.model = BertModel.from_pretrained(pretrained_model_name_or_path)

        # Get the hidden dimension of the last layer
        hidden = self.model.config.hidden_size

        self.fc = nn.Linear(hidden + self.include_label, self.num_classes)

    def forward(self, x, y=None):
        '''
            Use the cls token for prediction
            concatenate with the one hot label if y is provided (for the
            splitter only)
        '''
        input_ids = x[:, :, 0]
        input_masks = x[:, :, 1]
        segment_ids = x[:, :, 2]

        # Get the representation for the CLS token
        x = self.model(
            input_ids=input_ids,
            attention_mask=input_masks,
            token_type_ids=segment_ids,
        )[0][:,0,:]

        if self.include_label > 0:
            # Append the one hot label embedding
            x = torch.cat(
                [x, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )

        x = self.fc(x)

        return x
