import torch
import torch.nn as nn


def get_optim(model, cfg):
    '''
        return optimizer and scheduler (if for mnli)
    '''
    result = {
        'optim': getattr(torch.optim, cfg['optim']['name'])(
            filter(lambda p: p.requires_grad, model.parameters()),
            **cfg['optim']['args']
        )
    }

    if cfg['lr_scheduler']['name'] is not None:
        result['lr_scheduler'] = \
            getattr(torch.optim.lr_scheduler, cfg['lr_scheduler']['name'])(
                optimizer=result['optim'],
                **cfg['lr_scheduler']['args']
            )

    return result


def _valid_gradient(model: nn.Module):
    '''
        Check whether the gradients are valid (not nan)
    '''
    for p in model.parameters():
        if p.grad is not None:
            # Check whether all gradients in p are finite
            is_valid = torch.isfinite(p.grad).all()

            # The gradients are not valid if any of them is nan or inf
            if not is_valid:
                return False
    return True


def optim_step(model, opt_dict, loss, cfg):
    '''
        Apply the optimizer for one step. If the scheduler is present in the
        opt dict, update the scheduler after the optimizer.
    '''
    opt_dict['optim'].zero_grad()
    loss.backward()

    if not _valid_gradient(model):
        # Zero out the gradient if the gradients contain nan or inf
        opt_dict['optim'].zero_grad()

    else:

        if 'clip_grad_norm' in cfg['optim'] and cfg['optim']['clip_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg['optim']['clip_grad_norm'])

    opt_dict['optim'].step()

    if 'lr_scheduler' in opt_dict:
        opt_dict['lr_scheduler'].step()
