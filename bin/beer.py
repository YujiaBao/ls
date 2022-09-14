import ls


if __name__ == '__main__':

    print('Loading Beer Reviews')
    data = ls.datasets.BeerReviews(aspect='look')

    print('Learning to split')
    train_data, test_data = ls.learning_to_split(
        data,
        device='cuda:0',
        num_classes=2,
        model={
            'name': 'textcnn',
        },
        optim={
            # Support all optimizers under torch.optim
            'name': 'Adam',
            'args': {
                # Checkout torch.optim.Adam for more details.
                'lr': 0.001,
                'weight_decay': 0,
            },
        },
    )
