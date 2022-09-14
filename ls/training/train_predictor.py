import copy

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, random_split, RandomSampler, DataLoader

from ls.utils.optim import get_optim, optim_step
from ls.utils.print import print
from .test_predictor import test_predictor


def train_predictor(data: Dataset = None,
                    train_indices: list[int] = None,
                    predictor: torch.nn.Module = None,
                    cfg: dict = None):
    '''
        Train the predictor on the train split
        We sample a random set from train for early stopping and validation
    '''
    # Create the training split dataset by selecting the subset of the original
    # dataset.
    train_data = Subset(data, indices=train_indices)

    # Randomly sample a validation set (size 1/3) from the train split.
    # We will monitor the performance on the validation data to prevent
    # memorization.
    train_size = int(len(train_data) // 3 * 2)
    val_size   = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    train_loader = DataLoader(
        train_data, batch_size=cfg['batch_size'],
        sampler=RandomSampler(train_data, replacement=True,
                              num_samples=cfg['batch_size']*cfg['num_batches']),
        num_workers=cfg['num_workers']
    )

    val_loader = DataLoader(
        val_data, batch_size=cfg['batch_size'],
        sampler=RandomSampler(val_data, replacement=True,
                              num_samples=cfg['batch_size']*cfg['num_batches']),
        num_workers=cfg['num_workers'])

    # Get the optimizer of the predictor
    opt = get_optim(predictor, cfg)

    # Start training. Terminate training if the validation accuracy stops
    # improving.
    best_val_score = -1
    best_predictor = None
    ep, cycle = 0, 0

    while True:
        # train the predictor
        predictor.train()

        for x, y in train_loader:

            x = x.to(cfg['device'])
            y = y.to(cfg['device'])

            out = predictor(x)

            loss = F.cross_entropy(out, y)

            optim_step(predictor, opt, loss, cfg)

        val_acc = test_predictor(predictor=predictor, cfg=cfg,
                                 loader=val_loader)

        progress_message =  f'train predictor ep {ep}, val_acc {val_acc:.2f}'
        print(progress_message, end="\r", flush=True, time=True)

        if val_acc > best_val_score:
            best_val_score = val_acc
            best_predictor = copy.deepcopy(predictor.state_dict())
            cycle = 0
        else:
            cycle += 1

        if cycle == cfg['patience']:
            break

        ep += 1

    # Load the best predictor
    predictor.load_state_dict(best_predictor)

    # Clear the progress
    # Note that we need to overwrite the time stamps too.
    print(" " * (20 + len(progress_message)), end="\r", time=False)

    return best_val_score
