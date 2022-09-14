import ls


if __name__ == '__main__':
    print('Loading Waterbirds')
    data = ls.datasets.Waterbirds()

    print('Learning to split')
    train_data, test_data = ls.learning_to_split(
        data,
        # Overwriting configs/default.yaml
        device='cuda:0',
        num_classes=2,
        model={
            'name': 'resnet50',
        },
        optim={
            'name': 'SGD',
            # Clip the gradients if their norms exceed 1
            'clip_grad_norm': 1,
            # Checkout torch.optim.SGD for more details.
            'args': {
                'lr': 0.0001,
                'weight_decay': 0,
                'momentum': 0.9,
            },
        },
    )
