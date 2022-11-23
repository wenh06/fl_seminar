"""
"""

from typing import Union, Iterable

from torch.nn.parameter import Parameter

from .base import ProxSGD


__all__ = [
    "FedProxOptimizer",
]


class FedProxOptimizer(ProxSGD):
    """
    References
    ----------
    1. https://github.com/litian96/FedProx/blob/master/flearn/optimizer/pgd.py
    2. https://github.com/litian96/FedProx/blob/master/flearn/optimizer/pggd.py

    The `gold` (reference 2) is not re-implemented yet.
    """

    __name__ = "FedProxOptimizer"

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 1e-3,
        mu: float = 1e-2,
    ) -> None:
        """

        Parameters
        ----------
        params: iterable of dict or Parameter,
            the parameters to optimize
        lr: float, default 0.01,
            the learning rate
        mu: float, default 0.1,
            coeff. of the proximal term
        """
        self.mu = mu
        super().__init__(params, lr=lr, prox=mu, momentum=0)
