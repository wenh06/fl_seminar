"""
"""

from typing import Union, Iterable, NoReturn

from torch.nn.parameter import Parameter

from .base import ProxSGD


__all__ = ["FedDROptimizer",]


class FedDROptimizer(ProxSGD):
    """
    """
    __name__ = "FedDROptimizer"

    def __init__(self,
                 params:Iterable[Union[dict,Parameter]],
                 lr:float=1e-3, eta:float=1.0,) -> NoReturn:
        """

        Parameters
        ----------
        params: iterable of dict or Parameter,
            the parameters to optimize
        lr: float, default 0.01,
            the learning rate
        eta: float, default 0.1,
            reciprocal coeff. of the proximal term
        """
        self.eta = eta
        super().__init__(params, lr=lr, prox=1/eta, momentum=0)
