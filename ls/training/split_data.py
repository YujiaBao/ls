from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical

from ls.utils.print import print


def split_data(data: Dataset = None,
               splitter: torch.nn.Module = None,
               cfg: dict = None,
               random_split=False):
    '''
        Sample the splitting decisions from the Splitter.
        If random_split positive, apply random split instead.
    '''
    splitter.eval()

    # Load the data in testing mode (no shuffling so that we can keep track of
    # the index of each example)
    test_loader = DataLoader(data, shuffle=False,
                             batch_size=cfg['batch_size'],
                             num_workers=cfg['num_workers'])

    total_mask, total_y = [], []

    progress_message = 'split_data'
    print(progress_message, end="\r", flush=True, time=True)

    with torch.no_grad():
        for x, y in test_loader:

            # Move the current batch onto the device
            x = x.to(cfg['device'])
            y = y.to(cfg['device'])

            # Split each batch into train split and test split
            if random_split:
                # do random split at the start of ls
                prob = torch.ones_like(y).unsqueeze(1)
                prob = torch.cat([
                    prob * (1 - cfg['ratio']),  # test split prob
                    prob * cfg['ratio']  # train split prob
                ], dim=1)
            else:
                logit = splitter(x, y)
                prob = F.softmax(logit, dim=-1)

            # Sample the binary mask
            # 0: test split, 1: train split
            sampler = Categorical(prob)
            mask = sampler.sample()
            mask = mask.long()

            # save the sampling results
            total_mask.append(mask.cpu())
            total_y.append(y.cpu())

        # Aggregate all splitting decisions
        total_mask = torch.cat(total_mask)
        total_y = torch.cat(total_y)

    # Gather the training indices (indices with mask = 1)
    # and the testing indices (indices with mask = 0)
    train_indices = total_mask.nonzero().squeeze(1).tolist()
    test_indices = (1 - total_mask).nonzero().squeeze(1).tolist()

    # Compute the statistics of the current split
    split_stats = {}

    # Train/test size ratios
    split_stats['train_size'] = len(train_indices)
    split_stats['test_size'] = len(test_indices)
    split_stats['train_ratio'] = len(train_indices) / len(total_mask) * 100
    split_stats['test_ratio'] = len(test_indices) / len(total_mask) * 100

    # Label distribution
    split_stats['train_y'] = dict(sorted(
        Counter(total_y[train_indices].tolist()).items()))
    split_stats['test_y'] = dict(sorted(
        Counter(total_y[test_indices].tolist()).items()))

    print(" " * (20 + len(progress_message)), end="\r", time=False)

    return split_stats, train_indices, test_indices
