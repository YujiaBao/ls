import os
import json

import torch
from PIL import Image
from torchvision import transforms


class Waterbirds(torch.utils.data.Dataset):
    def __init__(self,
                 root: str = './datasets/waterbirds',
                 split: str = 'train'):

        if not os.path.isdir(root):
            # The Waterbirds data dir doesn't exist. Download the dataset.
            os.makedirs(root)
            Waterbirds.download(root)

        self.root = root
        self.image_paths, self.targets = Waterbirds.load_json(
            os.path.join(root, f'{split}.json'))

        self.length = len(self.targets)

        self.preprocessing = Waterbirds.get_transform_cub()

    @staticmethod
    def load_json(json_path: str):
        '''
            Read the json file that stores the image path and the label.
        '''
        image_paths, targets = [], []

        with open(json_path, 'r') as f:

            for line in f:
                example = json.loads(line)
                targets.append(example['y'])
                image_paths.append(example['x'])

        targets = torch.tensor(targets)

        return image_paths, targets

    @staticmethod
    def get_transform_cub():
        '''
            Transform the raw images so that it matches with the distribution of
            ImageNet
        '''
        scale = 256.0 / 224.0
        target_resolution = (224, 224)

        assert target_resolution is not None

        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale),
                               int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return transform

    @staticmethod
    def download(root: str):
        '''
            Download the Waterbirds dataset and the split information.
            The original dataset assumes a perfect validation split (each group
            has equal amount of annotations).
            We combine the train and valid data and perform a random split to
            get the training and validation data.
        '''
        os.system(
            "echo 'Downloading Waterbirds'\n"
            f"cd {root}\n"
            "wget https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz\n"
            "tar -xf waterbird_complete95_forest2water2.tar.gz\n"
            "wget https://people.csail.mit.edu/yujia/files/ls/waterbirds/train.json\n"
            "wget https://people.csail.mit.edu/yujia/files/ls/waterbirds/valid.json\n"
            "wget https://people.csail.mit.edu/yujia/files/ls/waterbirds/test.json\n"
            "mv waterbird_complete95_forest2water2/* ./\n"
            "rm -rf waterbird_complete95_forest2water2\n"
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        '''
            Given the index, return the bird image and the binary label.
        '''
        target = self.targets[idx]

        with Image.open(os.path.join(self.root, self.image_paths[idx])) as raw:
            img = self.preprocessing(raw)

        return img, target
