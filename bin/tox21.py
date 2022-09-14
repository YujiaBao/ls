import ls


if __name__ == '__main__':
    print("Loading Tox21 dataset")
    data = ls.datasets.Tox21()

    print("Learning to split")
    train_data, test_data = ls.learning_to_split(
        data,
        # Overwriting configs/default.yaml
        device='cuda:0',
        num_classes=2,
        metric='roc_auc',  # We use ROC instead of accuracy for Tox21
        model={
            'name': 'mlp',
            'args': {
                'hidden_dim_list': [1644, 1024, 1024, 1024],
                'dropout': 0.3
            },
        },
        optim={
            'name': 'Adam',
            'args': {
                'lr': 0.001,
                'weight_decay': 0,
            },
        },
        batch_size=200,
        num_batches=100,
    )
