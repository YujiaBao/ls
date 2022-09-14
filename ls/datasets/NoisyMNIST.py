import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torch.distributions.categorical import Categorical


class NoisyMNIST(Dataset):
    '''
        In NoisyMNIST, we inject random label noise into MNIST.
        Given a noise ratio r, we define the observed noisy label based on the
        following distribution:
            P(noisy_label = clean_label) = 1 - r
            P(noisy_label = y) = (1 - r) / 9 for any y != clean_label
    '''
    def __init__(self,
                 file_path: str = './datasets/mnist',
                 noise_ratio: float = 0.2):
        '''
            file_path: path to download/load the data
            noise_ratio: The probability that we purtub the true label.
        '''

        assert noise_ratio >= 0 and noise_ratio <= 1,\
            "Noise ratio has to be in the interval [0, 1]."

        mnist = MNIST(file_path, train=True, download=True)

        self.data = mnist.data.reshape((-1, 1, 28, 28)).float() / 255
        self.targets = NoisyMNIST.add_noise(mnist.targets, noise_ratio)

        self.length = len(mnist.data)

    def __len__(self):
        return self.length

    @staticmethod
    def add_noise(labels, noise_ratio):
        '''
            Inject noise to MNIST
        '''
        n_labels = 10

        # The probability that the noisy_label is equal to y (y != clean_label)
        # is noisy_ratio / (n_labels-1)
        prob_label = (torch.ones((n_labels, n_labels))
                      * (noise_ratio / (n_labels - 1)))

        # Set the probability that noisy_label == clean_label
        for i in range(n_labels):
            prob_label[i, i] = 1 - noise_ratio

        # Sample the noisy label for each example
        labels_prob = torch.index_select(prob_label, dim=0, index=labels)
        noisy_labels = Categorical(probs=labels_prob).sample()

        return noisy_labels.long()

    def __getitem__(self, idx):
        '''
            Return the image and the label for index idx
        '''
        return self.data[idx], self.targets[idx]

