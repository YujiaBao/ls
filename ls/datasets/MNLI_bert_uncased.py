import os
import csv

import torch

from ls.utils.print import print
# import ls.utils_glue as utils_glue


class MNLI_bert_uncased(torch.utils.data.Dataset):
    """
        MultiNLI dataset.
        label_dict = {
            'contradiction': 0,
            'entailment': 1,
            'neutral': 2
        }
    """
    def __init__(self,
                 root: str = './datasets/mnli',
                 split: str = 'train',):

        if not os.path.isdir(root):
            # The MNLI data dir doesn't exist. Download the dataset.
            os.makedirs(root)
            MNLI_bert_uncased.download(root)

        self.root = root
        assert split in ['train', 'val', 'test'], f"Unknown split {split}."

        self.data, self.targets = MNLI_bert_uncased.load_data(root, split)

        self.data = self.data[:100]
        self.targets = self.targets[:100]

        self.length = len(self.targets)

    @staticmethod
    def load_data(root: str, split: str):
        '''
            Load the dataset given the dataset root directory.
        '''

        # Step 1. Preparing the BERT data file
        data = []

        # Read all splits and combine them together
        for data_file in [
                'cached_train_bert-base-uncased_128_mnli',
                'cached_dev_bert-base-uncased_128_mnli',
                'cached_dev_bert-base-uncased_128_mnli-mm']:

            data += torch.load(os.path.join(root, data_file))

        # Gather all the input ids, masks, segements to create the BERT input
        # file
        input_ids = torch.tensor([f.input_ids for f in data], dtype=torch.long)

        input_masks = torch.tensor([f.input_mask for f in data],
                                   dtype=torch.long)

        segment_ids = torch.tensor([f.segment_ids for f in data],
                                   dtype=torch.long)

        label_ids = torch.tensor([f.label_id for f in data], dtype=torch.long)

        data = torch.stack((input_ids, input_masks, segment_ids), dim=2)

        # Step 2. Preparing the label / split information
        targets = []
        mask = []
        with open(os.path.join(root, 'metadata_random.csv')) as f:
            reader = csv.DictReader(f)

            for row in reader:

                targets.append(int(row['gold_label']))

                # If the example is within the given split, set mask = 1
                # Otherwise set mask = 0
                if split == 'train':
                    mask.append(row['split'] == '0')
                elif split == 'val':
                    mask.append(row['split'] == '1')
                elif split == 'test':
                    mask.append(row['split'] == '2')
                else:
                    ValueError(f"Unknown split {split}")

        targets = torch.tensor(targets).long()
        mask = torch.tensor(mask).bool()

        # Verify that the csv file aligns with the precomputed features
        assert torch.equal(targets, label_ids), \
            "Preprocessed data does not align with the csv file."

        # Return only the selected examples
        return data[mask], targets[mask]

    @staticmethod
    def download(root: str):
        '''
            Download the preprocessed dataset provided kindly provided by
            https://github.com/kohpangwei/group_DRO
        '''
        os.system(
            "echo 'Downloading MNLI'\n"
            f"cd {root}\n"
            "wget https://nlp.stanford.edu/data/dro/multinli_bert_features.tar.gz\n"
            "tar -xf multinli_bert_features.tar.gz\n"
            "mv multinli_bert_features/* ./\n"
            "wget https://github.com/kohpangwei/group_DRO/raw/master/dataset_metadata/multinli/metadata_random.csv\n"
            "rm -rf multinli_bert_features\n"
            "rm multinli_bert_features.tar.gz\n"
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
