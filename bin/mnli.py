import ls


def lr_lambda(current_step: int):
    '''
        A scaling factor for the learning rate.
        https://github.com/huggingface/transformers/blob/v4.21.3/src/transformers/optimization.py#L75
    '''
    num_warmup_steps = 0
    num_training_steps = 100000
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    return max(0.0, float(num_training_steps - current_step) /
               float(max(1, num_training_steps- num_warmup_steps)))


if __name__ == '__main__':

    print('Loading MNLI')
    data = ls.datasets.MNLI_bert_uncased()

    print('Learning to split')
    train_data, test_data = ls.learning_to_split(
        data,
        device='cuda:0',
        num_classes=3,  # entailment, neutral, contradiction
        model={
            'name': 'bert',
        },
        optim={
            'name': 'AdamW',
            'clip_grad_norm': 1,
            'args': {
                'lr': 0.00002,
                'weight_decay': 0,
                'eps': 1.0e-8,
            },
        },
        lr_scheduler={
            # Check torch.optim.lr_scheduler for more details.
            'name': 'LambdaLR',
            'args': {
                'lr_lambda': lr_lambda,
            }
        },
    )
