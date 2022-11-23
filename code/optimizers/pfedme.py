"""
"""

from typing import Iterable, Union, Optional, Tuple

import torch  # noqa: F401
from torch import Tensor
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer

from .base import ProxSGD


__all__ = [
    "pFedMeOptimizer",
]


class pFedMeOptimizer(ProxSGD):
    """ """

    __name__ = "pFedMeOptimizer"

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 0.01,
        lamda: float = 0.1,
        mu: float = 1e-3,
    ) -> None:
        """ """
        self.lamda = lamda
        self.mu = mu
        super().__init__(params, lr=lr, prox=lamda, momentum=mu, nesterov=True)

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", True)


class _pFedMeOptimizer(Optimizer):
    """legacy pFedMeOptimizer"""

    __name__ = "_pFedMeOptimizer"

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 0.01,
        lamda: float = 0.1,
        mu: float = 1e-3,
    ) -> None:
        """

        Parameters
        ----------
        params: iterable of dict or Parameter,
            the parameters to optimize
        lr: float, default 0.01,
            the learning rate
        lamda: float, default 0.1,
            coeff. of the proximal term
        mu: float, default 1e-3,
            momentum coeff.
        """
        # self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super().__init__(params, defaults)

    def step(
        self,
        local_weight_updated: Iterable[Parameter],
        closure: Optional[callable] = None,
    ) -> Tuple[Iterable[Parameter], Optional[Tensor]]:
        """

        Parameters
        ----------
        local_weight_updated: iterable of Parameter,
            the local weights updated by the server
        closure: callable, optional,
            a closure that reevaluates the model and returns the loss.

        Returns
        -------
        group["params"]: list of Parameter,
            the list of Tensors for the updated parameters
        loss: Tensor, optional,
            the loss after the step
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p, localweight in zip(group["params"], local_weight_updated):
                p.data = p.data - group["lr"] * (
                    p.grad.data
                    + group["lamda"] * (p.data - localweight.data)
                    + group["mu"] * p.data
                )
        return group["params"], loss

    def update_param(
        self,
        local_weight_updated: Iterable[Parameter],
        closure: Optional[callable] = None,
    ) -> Iterable[Parameter]:
        """

        Parameters
        ----------
        local_weight_updated: iterable of Parameter,
            the local weights updated by the server
        closure: callable, optional,
            a closure that reevaluates the model and returns the loss.

        Returns
        -------
        group["params"]: list of Parameter,
            the list of Tensors for the updated parameters
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p, localweight in zip(group["params"], local_weight_updated):
                p.data = localweight.data.clone()
        # return  p.data
        return group["params"]


# -----------------------------
# the following for comparison


class FEDLOptimizer(Optimizer):
    """ """

    __name__ = "FEDLOptimizer"

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 0.01,
        server_grads: Optional[Tensor] = None,
        pre_grads: Optional[Tensor] = None,
        eta: float = 0.1,
    ) -> None:
        """

        Parameters
        ----------
        to write
        """
        self.server_grads = server_grads
        self.pre_grads = pre_grads
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, eta=eta)
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None) -> Optional[Tensor]:
        """

        Parameters
        ----------
        closure: callable, optional,
            a closure that reevaluates the model and returns the loss

        Returns
        -------
        loss: Tensor, optional,
            the loss after the step
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            i = 0
            for p in group["params"]:
                p.data = p.data - group["lr"] * (
                    p.grad.data
                    + group["eta"] * self.server_grads[i]
                    - self.pre_grads[i]
                )
                # p.data.add_(p.grad.data, alpha=-group["lr"])
                i += 1
        return loss


class APFLOptimizer(Optimizer):
    """ """

    __name__ = "APFLOptimizer"

    def __init__(
        self, params: Iterable[Union[dict, Parameter]], lr: float = 0.01
    ) -> None:
        """ """
        defaults = dict(lr=lr)
        super(APFLOptimizer, self).__init__(params, defaults)

    def step(
        self, closure: Optional[callable] = None, beta: float = 1.0, n_k: float = 1.0
    ) -> Optional[Tensor]:
        """ """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # print(group)
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = beta * n_k * p.grad.data
                p.data.add_(d_p, alpha=-group["lr"])
        return loss
