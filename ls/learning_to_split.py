import copy
from datetime import datetime

import torch
from torch.utils.data import Dataset, Subset

from . import utils
from .utils.print import print
from .training import split_data, train_predictor, test_predictor, train_splitter
from .models.build import ModelFactory


def print_split_status(outer_loop: int,
                       split_stats: dict,
                       val_score: float,
                       test_score: float):
    '''
        Print the stats of the current split.
    '''
    print(f"[red][bold]ls outer loop {outer_loop}[/bold][/red] @ "
          f"[not bold][default]"
          f"{datetime.now().strftime('%H:%M:%S %Y/%m/%d')}"
          f"[/not bold][default]")

    print(f"| generalization gap {val_score-test_score:>5.2f} "
          f"(val {val_score:>5.2f}, test {test_score:>5.2f})")

    print(f"| train count {split_stats['train_ratio']:4.1f}% "
          f"({split_stats['train_size']:d})")
    print(f"| test  count {split_stats['test_ratio']:4.1f}% "
          f"({split_stats['test_size']:d})")

    print(f"| train label dist {split_stats['train_y']}")
    print(f"| test  label dist {split_stats['test_y']}")


def learning_to_split(data: Dataset,
                      return_order: list[str] = ['train_data', 'test_data'],
                      verbose: bool = True,
                      **overwrite_config):
    '''
        ls: learning to split
        train a splitter to split the dataset into training/testing
        such that a predictor learned on training cannot generalize to testing
    '''
    cfg = utils.read_config(overwrite_config)

    num_no_improvements = 0

    best_gap = -1  # The bigger the gap, the better (more challenging) the split.
    best_split = None

    splitter = ModelFactory.get_model(cfg, splitter=True)

    opt = utils.optim.get_optim(splitter, cfg)

    for outer_loop in range(cfg['num_outer_loop']):

        #### STEP 1 ####
        # Split the dataset using the Splitter
        # We start with random split for the first iteration
        random_split = True if outer_loop == 0 else False
        split_stats, train_indices, test_indices = split_data(
            data, splitter, cfg, random_split)

        #### STEP 2 ####
        # Train the Predictor (from scratch) on the train split
        # (using a subset of train split for validation -- early stopping)
        #
        # Get the predictor.
        predictor = ModelFactory.get_model(cfg, predictor=True)
        #
        # Train it on the training split
        val_score = train_predictor(data=data, train_indices=train_indices,
                                    predictor=predictor, cfg=cfg)
        #
        # Evaluate the train (validation) / test gap and print the split stats.
        test_score = test_predictor(data=data, test_indices=test_indices,
                                    predictor=predictor, cfg=cfg)

        if verbose:
            print_split_status(outer_loop, split_stats, val_score, test_score)

        # Save the splitter if it produces a more challenging split
        gap = val_score - test_score
        if gap > best_gap:
            best_gap = gap
            num_no_improvements = 0
            best_split = {
                'splitter':      copy.deepcopy(splitter.state_dict()),
                'train_indices': train_indices,
                'test_indices':  test_indices,
                'val_score':     val_score,
                'test_score':    test_score,
                'split_stats':   split_stats,
                'outer_loop':    outer_loop
            }
        else:
            num_no_improvements += 1

        if num_no_improvements == cfg['patience']:
            break

        #### STEP 3 ####
        # Train the splitter to create a more challenging split based on the
        # predictor's performance
        train_splitter(splitter, predictor, data, test_indices, opt, cfg,
                       verbose=verbose)

    # Done! Print the best split.
    if verbose:
        print("Finished!\nBest split:")
        print_split_status(best_split['outer_loop'], best_split['split_stats'],
                           best_split['val_score'], best_split['test_score'])

    # Preparing the output
    outputs = []
    for element in return_order:

        if element == 'train_data':
            # return the training split
            outputs.append(Subset(data, indices=best_split['train_indices']))

        elif element == 'test_data':
            # return the testing split
            outputs.append(Subset(data, indices=best_split['test_indices']))

        elif element == 'train_indices':
            # return the indicies of the training split examples
            outputs.append(best_split['train_indices'])

        elif element == 'test_indices':
            # return the indicies of the testing split examples
            outputs.append(best_split['test_indices'])

        elif element == 'splitter':
            # return the learned splitter
            splitter.load_state_dict(best_split['splitter'])
            outputs.append(splitter)

        else:
            raise ValueError(f'Unsupported return type {element}')

    return tuple(outputs)
