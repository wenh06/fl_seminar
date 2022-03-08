"""
"""

from abc import ABC, abstractmethod
from typing import Iterable, Any, Union, NoReturn

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.optim import Optimizer

from ..optimizers import get_optimizer as get_optimizer_


__all__ = [
    "get_optimizer",
]


def get_optimizer(optimizer_name:str, params:Iterable[Union[dict,Parameter]], config:Any) -> Optimizer:
    """
    """
    try:
        return get_optimizer_(optimizer_name, params, config)
    except:
        raise NotImplementedError
