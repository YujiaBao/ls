import torch
import torch.nn.functional as F


def compute_marginal_z_loss(mask, tar_ratio, no_grad=False):
    '''
        Compute KL div between the splitter's z marginal and the prior z margional
        Goal: the predicted training size need to be tar_ratio * total_data_size
    '''
    cur_ratio = torch.mean(mask)
    cur_z = torch.stack([cur_ratio, 1 - cur_ratio])  # train split, test_split

    tar_ratio = torch.ones_like(cur_ratio) * tar_ratio
    tar_z = torch.stack([tar_ratio, 1 - tar_ratio])

    loss_ratio = F.kl_div(torch.log(cur_z), tar_z, reduction='batchmean')

    if not torch.isfinite(loss_ratio):
        loss_ratio = torch.ones_like(loss_ratio)

    if no_grad:
        loss_ratio = loss_ratio.item()

    return loss_ratio, cur_ratio.item()


def compute_y_given_z_loss(mask, y, no_grad=False):
    '''
      conditional marginal p(y | z = 1) need to match p(y | z = 0)
    '''
    # get num of classes
    num_classes = len(torch.unique(y))

    y_given_train, y_given_test, y_original = [], [], []

    for i in range(num_classes):
        y_i = (y == i).float()

        y_given_train.append(torch.sum(y_i * mask) / torch.sum(mask))
        y_given_test.append(torch.sum(y_i * (1 - mask)) / torch.sum(1 - mask))
        y_original.append(torch.sum(y_i) / len(y))

    y_given_train = torch.stack(y_given_train)
    y_given_test = torch.stack(y_given_test)
    y_original = torch.stack(y_original).detach()

    loss_y_marginal = F.kl_div(torch.log(y_given_train), y_original,
                               reduction='batchmean') + \
        F.kl_div(torch.log(y_given_test), y_original, reduction='batchmean')

    if not torch.isfinite(loss_y_marginal):
        loss_y_marginal = torch.ones_like(loss_y_marginal)

    if no_grad:
        loss_y_marginal = loss_y_marginal.item()

    return loss_y_marginal, y_given_train.tolist()[-1], y_given_test.tolist()[-1]

