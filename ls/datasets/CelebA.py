from typing import Optional, Callable

import torch
from torchvision import datasets, transforms


class CelebA(torch.utils.data.Dataset):
    def __init__(self,
                 root: str = './datasets/celeba',
                 split: str = 'train',
                 target_type: str = 'attr',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = True
                 ):
        '''
            This dataset is basically the same as torchvision.datasets.CelebA
            except that we return the image and the label (blond hair vs. not
            blond hair) in  __getitem__()

            See
            https://pytorch.org/vision/main/generated/torchvision.datasets.CelebA.html
            for more details.
        '''
        # Load the CelebA data
        self.data = datasets.CelebA(root,
                                    split,
                                    target_type,
                                    transform,
                                    target_transform,
                                    download)

        # get the idx of label
        self.label_idx = self.data.attr_names.index('Blond_Hair')

        # ImageNet transformation
        self.preprocessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Use the original torchvision api to get the image and the attributes
        img, attrs = self.data[idx]

        # Get the target label (blond hair or not)
        target = attrs[self.label_idx]

        # Preprocess the image so that it can be used by pre-trained models
        img = self.preprocessing(img)

        return img, target
