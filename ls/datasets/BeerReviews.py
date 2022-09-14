import os
import json

import torch
import torch.nn.functional as F
import torchtext


class BeerReviews(torch.utils.data.Dataset):
    def __init__(self,
                 root: str = './datasets/beer',
                 aspect: str = 'look'):
        '''
            Load the beer review dataset for the given aspect.
        '''
        assert aspect in ["look", "aroma"], f"Unknown aspect {aspect}."

        if not os.path.isdir(root):
            # The tox21 data dir doesn't exist. Download the dataset.
            os.makedirs(root)
            BeerReviews.download(root)

        # get the data and binary targets
        self.data, self.targets = BeerReviews.load_json(root, aspect)

        # Set the maximum sequence length
        self.max_seq_len = max([len(text) for text in self.data])

        self.length = len(self.targets)

        # get word embeddings from fasttext
        self.vocab = BeerReviews.get_vocab('FastText')

    @staticmethod
    def download(root: str):
        '''
            Download the beer review dataset.
        '''
        os.system(
            "echo 'Downloading BeerReviews'\n"
            f"cd {root}\n"
            "wget http://people.csail.mit.edu/yujia/files/ls/beer/look.json\n"
            "wget http://people.csail.mit.edu/yujia/files/ls/beer/aroma.json\n"
        )

    @staticmethod
    def load_json(root: str, aspect: str):
        '''
            Load the data json file.
        '''
        data, targets = [], []

        path = os.path.join(root, aspect + '.json')

        with open(path, 'r') as f:
            for line in f:
                example = json.loads(line)
                targets.append(example['y'])
                data.append(example['text'])

        targets = torch.tensor(targets)

        return data, targets

    @staticmethod
    def get_vocab(name: str = 'FastText'):
        '''
            Get the pre-trained vocab.
        '''
        vocab = getattr(torchtext.vocab, name)()

        # Add the pad token
        specials = ['<pad>']
        for token in specials:
            vocab.stoi[token] = len(vocab.itos)
            vocab.itos.append(token)

        vocab.vectors = torch.cat(
            [vocab.vectors, torch.zeros(1, 300)], dim=0)

        return vocab

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        '''
            Retrieve the text and label for the given index. Pad the input so
            that they all have equal length.
        '''
        text = self.data[idx]

        padded_text = self.vocab.stoi['<pad>'] * torch.ones(self.max_seq_len)
        padded_text[:len(text)] = torch.tensor([
            self.vocab.stoi[token] if token in self.vocab.stoi
            else self.vocab.stoi['unk'] for token in text
        ])

        # seq_len x embedding_dim
        feat = F.embedding(padded_text.long(), self.vocab.vectors).detach()

        # Transpose it for CNN
        # embedding_dim x seq_len
        feat = feat.contiguous().transpose(0, 1)

        return feat, self.targets[idx]
