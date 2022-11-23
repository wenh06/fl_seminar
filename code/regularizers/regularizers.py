"""
this file is forked from https://github.com/unc-optimization/FedDR/tree/main/FedDR/flearn/regularizers
"""

import re
from abc import ABC, abstractmethod
from math import sqrt
from typing import Iterable, List, Optional

import torch  # noqa: F401
from torch.nn.parameter import Parameter

from misc import ReprMixin


__all__ = [
    "get_regularizer",
    "Regularizer",
    "L1Norm",
    "L2Norm",
    "L2NormSquared",
    "LInfNorm",
    "NullRegularizer",
]


class Regularizer(ReprMixin, ABC):
    """ """

    __name__ = "Regularizer"

    def __init__(self, coeff: float = 1.0) -> None:
        """ """
        self.coeff = coeff

    @abstractmethod
    def eval(self, params: Iterable[Parameter], coeff: Optional[float] = None) -> float:
        """ """
        raise NotImplementedError

    @abstractmethod
    def prox_eval(
        self, params: Iterable[Parameter], coeff: Optional[float] = None
    ) -> Iterable[Parameter]:
        """ """
        raise NotImplementedError

    def extra_repr_keys(self) -> List[str]:
        """ """
        return super().extra_repr_keys() + [
            "coeff",
        ]


def get_regularizer(reg_type: str, reg_coeff: float = 1.0) -> Regularizer:
    """ """
    reg_type = re.sub("regularizer|norm|[\\s\\_\\-]+", "", reg_type.lower())
    if reg_type in [
        "l1",
    ]:
        return L1Norm(reg_coeff)
    elif reg_type in [
        "l2",
    ]:
        return L2Norm(reg_coeff)
    elif reg_type in [
        "l2squared",
    ]:
        return L2NormSquared(reg_coeff)
    elif reg_type in [
        "no",
        "empty",
        "zero",
        "none",
        "null",
    ]:
        return NullRegularizer(reg_coeff)
    elif reg_type in [
        "linf",
        "inf",
        "linfinity",
        "infinity",
        "linfty",
        "infty",
    ]:
        return LInfNorm(reg_coeff)
    else:
        raise ValueError(f"Unknown regularizer type: {reg_type}")


class NullRegularizer(Regularizer):
    """null regularizer, or equivalently the zero function"""

    __name__ = "NullRegularizer"

    def eval(self, params: Iterable[Parameter], coeff: Optional[float] = None) -> float:
        """ """
        return 0.0

    def prox_eval(
        self, params: Iterable[Parameter], coeff: Optional[float] = None
    ) -> Iterable[Parameter]:
        """ """
        return list(params)


class L1Norm(Regularizer):
    """ """

    __name__ = "L1Norm"

    def eval(self, params: Iterable[Parameter], coeff: Optional[float] = None) -> float:
        """ """
        if coeff is None:
            coeff = self.coeff
        return coeff * sum([p.data.abs().sum().item() for p in params])

    def prox_eval(
        self, params: Iterable[Parameter], coeff: Optional[float] = None
    ) -> Iterable[Parameter]:
        """ """
        if coeff is None:
            coeff = self.coeff
        ret_params = [
            p.data.sign() * (p.data.abs() - coeff).clamp(min=0) for p in params
        ]
        return ret_params


class L2Norm(Regularizer):
    """ """

    __name__ = "L2Norm"

    def eval(self, params: Iterable[Parameter], coeff: Optional[float] = None) -> float:
        """ """
        if coeff is None:
            coeff = self.coeff
        return coeff * sqrt(sum([p.data.pow(2).sum().item() for p in params]))

    def prox_eval(
        self, params: Iterable[Parameter], coeff: Optional[float] = None
    ) -> Iterable[Parameter]:
        """ """
        if coeff is None:
            coeff = self.coeff
        _params = list(params)  # to avoid the case that params is a generator
        norm = self.eval(_params, coeff=coeff)
        coeff = max(0, 1 - coeff / norm)
        ret_params = [coeff * p.data for p in _params]
        del _params
        return ret_params


class L2NormSquared(Regularizer):
    """ """

    __name__ = "L2NormSquared"

    def eval(self, params: Iterable[Parameter], coeff: Optional[float] = None) -> float:
        """ """
        if coeff is None:
            coeff = self.coeff
        return coeff * sum([p.data.pow(2).sum().item() for p in params])

    def prox_eval(
        self, params: Iterable[Parameter], coeff: Optional[float] = None
    ) -> Iterable[Parameter]:
        """ """
        if coeff is None:
            coeff = self.coeff
        coeff = 1 / (1 + 2 * coeff)
        _params = list(params)  # to avoid the case that params is a generator
        ret_params = [coeff * p.data for p in _params]
        del _params
        return ret_params


class LInfNorm(Regularizer):
    """ """

    __name__ = "LInfNorm"

    def eval(self, params: Iterable[Parameter], coeff: Optional[float] = None) -> float:
        """ """
        if coeff is None:
            coeff = self.coeff
        return coeff * max([p.data.abs().max().item() for p in params])

    def prox_eval(
        self, params: Iterable[Parameter], coeff: Optional[float] = None
    ) -> Iterable[Parameter]:
        """ """
        if coeff is None:
            coeff = self.coeff
        _params = list(params)  # to avoid the case that params is a generator
        raise NotImplementedError("L-infinity norm is not implemented yet")
