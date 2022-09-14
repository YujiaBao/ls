import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.metrics import roc_auc_score


def test_predictor(data: Dataset = None,
                   loader: DataLoader = None,
                   test_indices: list[int] = None,
                   predictor: torch.nn.Module = None,
                   cfg: dict = None,
                   percentage_score: bool = True):
    '''
        Apply the predictor to the test loader
    '''
    predictor.eval()

    # If the data loader is not provided, create a data loader from the data.
    if loader is None:
        assert data is not None, "data and loader cannot both be None"

        if test_indices is None:
            # Use the entire dataset for evaluation.
            test_data = data

        else:
            # Create the testing split dataset by selecting the subset of the
            # original dataset.
            test_data = Subset(data, indices=test_indices)

        loader = DataLoader(test_data, batch_size=cfg['batch_size'],
                            shuffle=False, num_workers=cfg['num_workers'])

    with torch.no_grad():

        total = {
            'correct': [],
            'pred_score': [],
            'true': [],
        }

        for x, y in loader:

            x = x.to(cfg['device'])
            y = y.to(cfg['device'])

            out = predictor(x)

            if cfg['metric'] == 'accuracy':
                total['correct'].append((torch.argmax(out, dim=1) == y).cpu())

            elif cfg['metric'] == 'roc_auc':
                # Assume that y = 1 is positive
                total['pred_score'].append(out[:, 1].cpu())
                total['true'].append(y.cpu())

            else:
                raise ValueError(f"Unknown metric type {cfg['metric']}")

        if cfg['metric'] == 'accuracy':
            score = torch.mean(torch.cat(total['correct']).float()).item()

        elif cfg['metric'] == 'roc_auc':
            true = torch.cat(total['true']).tolist()
            pred_score = torch.cat(total['pred_score']).tolist()
            score = roc_auc_score(true, pred_score)

        else:
            raise ValueError(f"Unknown metric type {cfg['metric']}")

    if percentage_score:
        # Return the score in percentage format.
        return score * 100
    else:
        return score
