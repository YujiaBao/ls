from typing import Callable

import torch.nn as nn


class ModelFactory:
    """ The factory class for creating models"""

    registry = {}
    """ Internal registry for available models """

    @classmethod
    def register(cls, name: str) -> Callable:
        """ Class method to register Executor class to the internal registry.

        Args:
            name (str): The name of the executor.

        Returns:
            The model class itself.
        """

        def inner_wrapper(wrapped_class: nn.Module) -> Callable:
            if name in cls.registry:
                print(f'Model {name} already exists. Will replace it')

            cls.registry[name] = wrapped_class

            return wrapped_class

        return inner_wrapper

    @classmethod
    def get_model(cls, cfg: dict, splitter=False, predictor=False) -> nn.Module:
        '''
            Return a model (nn.Module) based on the input:
                cfg: configuration file
                splitter: whether this is the splitter or not
                predictor: whether this is the predictor or not
        '''
        if cfg['model']['name'] not in cls.registry:
            raise ValueError(
                f"Model {cfg['model']['name']} does not exist in the registry")

        assert (int(splitter) + int(predictor)) == 1,\
            "Either splitter or predictor must be true."

        exec_class = cls.registry[cfg['model']['name']]

        if splitter:
            model = exec_class(
                # append num_classes to the feature representation
                include_label = cfg['num_classes'],
                # 2 outputs: in_train or in_test
                num_classes = 2,
                **cfg['model']['args'])
        else:
            # This is the predictor.
            model = exec_class(
                # do not append the ground truth label
                include_label = 0,
                # predict the original label
                num_classes = cfg['num_classes'],
                **cfg['model']['args'])

        return model.to(cfg['device'])

