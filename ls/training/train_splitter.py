import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, RandomSampler, DataLoader

from ls.utils.optim import optim_step
from ls.utils.print import print
from .utils import compute_marginal_z_loss, compute_y_given_z_loss


def print_epoch_res(stats, i):
    print(f"| splitter ep {i} "
          f"loss {stats['loss']:>6.4f} "
          f"(gap {stats['loss_gap']:>6.4f} "
          f"ratio {stats['loss_ratio']:>6.4f} "
          f"label {stats['loss_balance']:>6.4f})",
          flush=True)


def train_splitter_one_epoch(splitter, predictor, total_loader, test_loader,
                             opt, cfg):
    '''
        train the splitter for one epoch
    '''
    stats = {}
    for k in ['loss_ratio', 'loss_balance', 'loss_gap', 'loss', 'ratio',
              'ptrain_y', 'ptest_y']:
        stats[k] = []

    for batch_total, batch_test in zip(total_loader, test_loader):

        # regularity constraints
        x = batch_total[0].to(cfg['device'])
        y = batch_total[1].to(cfg['device'])

        logit = splitter(x, y)
        prob = F.softmax(logit, dim=-1)[:, 1]

        # This loss ensures that the training size and testing size are
        # compariable.
        loss_ratio, ratio = compute_marginal_z_loss(prob, cfg['ratio'])

        # This loss ensures that the marginal distributions of the label are
        # compariable in the training split and in the testing split
        loss_balance, ptrain_y, ptest_y = compute_y_given_z_loss(prob, y)

        stats['loss_ratio'].append(loss_ratio.item())
        stats['loss_balance'].append(loss_balance.item())

        # Evaluate generalization gap loss over the testing loader
        x = batch_test[0].to(cfg['device'])
        y = batch_test[1].to(cfg['device'])

        # Compare the splitter's prob vs. the predictor's prediction correctness
        logit = splitter(x, y)
        with torch.no_grad():
            out = predictor(x)
            correct = (torch.argmax(out, dim=1) == y).long()

        loss_gap = F.cross_entropy(logit, correct)
        stats['loss_gap'].append(loss_gap.item())

        # compute overall loss and update the parameters
        w_sum = cfg['w_gap'] + cfg['w_ratio'] + cfg['w_balance']
        loss = (loss_gap * cfg['w_gap']
                + loss_ratio * cfg['w_ratio']
                + loss_balance * cfg['w_balance']) / w_sum
        stats['loss'].append(loss.item())

        optim_step(splitter, opt, loss, cfg)

    for k, v in stats.items():
        stats[k] = torch.mean(torch.tensor(v)).item()

    return stats


def train_splitter(splitter: nn.Module,
                   predictor: nn.Module,
                   data: Dataset,
                   test_indices: list[int],
                   opt: dict,
                   cfg: dict,
                   verbose: bool = False):
    '''
        Train the splitter to
        1. (on the test split) move the correct predictions back to the train
        split and keep the incorrect predictions in the test split;
        2. (on all data) satisfy the regularity constraints
    '''
    splitter.train()
    predictor.eval()

    # total_loader samples from the entire dataset
    # We use its samples to enforce the regularity constraints
    total_loader = DataLoader(
        data, batch_size=cfg['batch_size'],
        sampler=RandomSampler(data, replacement=True,
                              num_samples=cfg['batch_size']*cfg['num_batches']),
        num_workers=cfg['num_workers']
    )

    # test_loader samples only from the testing split
    # We use its samples to learn how to create a challening split
    test_data = Subset(data, indices=test_indices)
    test_loader = DataLoader(
        test_data, batch_size=cfg['batch_size'],
        sampler=RandomSampler(test_data, replacement=True,
                              num_samples=cfg['batch_size']*cfg['num_batches']),
        num_workers=cfg['num_workers']
    )

    loss_list = []  # Keep track of the loss over the past 5 epochs
    ep = 0

    while True:
        train_stats = train_splitter_one_epoch(
            splitter, predictor, total_loader, test_loader, opt, cfg)

        cur_loss = train_stats['loss']
        if len(loss_list) == 5:
            if cur_loss > sum(loss_list) / 5.0 - cfg['convergence_thres']:
                # break if the avg loss in the past 5 iters doesn't improve a
                # lot
                break

            loss_list.pop(0)

        loss_list.append(cur_loss)

        progress_message =  f'train_splitter ep {ep}, loss {cur_loss:.4f}'
        print(progress_message, end="\r", flush=True, time=True)

        ep += 1
        if ep == 100:
            warnings.warn("Splitter (inner-loop) training fails to converge"
                          " within 100 epochs. ")

        elif ep == 1000:
            raise Error("Splitter (inner-loop) training fails to converge"
                        "within 1000 epochs.")

    # Clear the progress
    # Note that we need to overwrite the time stamps too.
    print(" " * (20 + len(progress_message)), end="\r", time=False)

    # Print the last status
    print_epoch_res(train_stats, ep)
