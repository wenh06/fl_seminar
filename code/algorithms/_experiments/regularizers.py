"""
this file is forked from https://github.com/unc-optimization/FedDR/tree/main/FedDR/flearn/regularizers
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from math import sqrt
from typing import Iterable, List, NoReturn, Optional

import torch
from torch.nn.parameter import Parameter

from misc import ReprMixin


__all__ = [
    "Regularizer", "get_regularizer",
    "L1Norm", "L2Norm",
]


class Regularizer(ReprMixin, ABC):
    """
    """
    __name__ = "Regularizer"

    def __init__(self, coeff:float=1.0) -> NoReturn:
        """
        """
        self.coeff = coeff

    @abstractmethod
    def eval(self, params:Iterable[Parameter], coeff:Optional[float]=None) -> float:
        """
        """
        raise NotImplementedError

    @abstractmethod
    def prox_eval(self, params:Iterable[Parameter], coeff:Optional[float]=None) -> Iterable[Parameter]:
        """
        """
        raise NotImplementedError

    def extra_repr_keys(self) -> List[str]:
        """
        """
        return super().extra_repr_keys() + ["coeff",]


def get_regularizer(reg_type:str, reg_coeff:float=1.0) -> Regularizer:
    """
    """
    if reg_type.lower() in ["l1", "l1_norm", "l1norm",]:
        return L1Norm(reg_coeff)
    elif reg_type.lower() in ["l2", "l2_norm", "l2norm",]:
        return L2Norm(reg_coeff)
    else:
        raise ValueError("Unknown regularizer type: {}".format(reg_type))


class L1Norm(Regularizer):
    """
    """
    __name__ = "L1Norm"

    def eval(self, params:Iterable[Parameter], coeff:Optional[float]=None) -> float:
        """
        """
        if coeff is None:
            coeff = self.coeff
        return coeff * sum([p.data.abs().sum().item() for p in params])

    def prox_eval(self, params:Iterable[Parameter], coeff:Optional[float]=None) -> Iterable[Parameter]:
        """
        """
        if coeff is None:
            coeff = self.coeff
        ret_params = [
            p.data.sign() * (p.data.abs()-coeff).clamp(min=0) \
                for p in params
        ]
        return ret_params


class L2Norm(Regularizer):
    """
    """
    __name__ = "L2Norm"

    def eval(self, params:Iterable[Parameter], coeff:Optional[float]=None) -> float:
        """
        """
        if coeff is None:
            coeff = self.coeff
        return coeff * sqrt(sum([p.data.pow(2).sum().item() for p in params]))

    def prox_eval(self, params:Iterable[Parameter], coeff:Optional[float]=None) -> Iterable[Parameter]:
        """
        """
        if coeff is None:
            coeff = self.coeff
        _params = list(params)  # to avoid the case that params is a generator
        norm = self.eval(_params, coeff=coeff)
        ret_params = [
            max(0, 1 - coeff / norm) * p.data \
                for p in _params
        ]
        del _params
        return ret_params
