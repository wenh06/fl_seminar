"""
"""

from typing import Iterable, Union, NoReturn, Optional, Tuple, List

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer


__all__ = [
    "ProxSGD", "prox_sgd",
]


class ProxSGD(Optimizer):
    """
    Proximal Stochastic Gradient Descent
    """
    __name__ = "ProxSGD"

    def __init__(self,
                 params:Iterable[Union[dict,Parameter]],
                 lr:float=1e-3,
                 momentum=1e-3, dampening=0,
                 weight_decay=0, nesterov=False,
                 prox:float=0.1,) -> NoReturn:
        """

        Parameters
        ----------
        params: iterable of dict or Parameter,
            the parameters to optimize
        lr: float, default 1e-3,
            the learning rate
        momentum: float, default 1e-3,
            momentum factor
        dampening: float, default 0,
            dampening for momentum
        weight_decay: float, default 0
            weight decay (L2 penalty)
        nesterov: bool, default False,
            if True, enables Nesterov momentum
        prox: float, default 0.1,
            coeff. of the proximal term
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, prox=prox)
        super().__init__(params, defaults)

    def __setstate__(self, state:dict) -> NoReturn:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self,
             local_weight_updated:Iterable[Parameter],
             closure:Optional[callable]=None,) -> Optional[Tensor]:
        """

        Parameters
        ----------
        local_weight_updated: iterable of Parameter,
            the local weights updated by the local optimizer,
            or of the previous iteration
        closure: callable, optional,
            a closure that reevaluates the model and returns the loss.

        Returns
        -------
        loss: Tensor, optional,
            the loss after the step
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            lr = group["lr"]
            prox = group["prox"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

            prox_sgd(
                params_with_grad, local_weight_updated,
                d_p_list, momentum_buffer_list,
                weight_decay, momentum, lr, dampening, nesterov, prox,
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss


def prox_sgd(params: List[Tensor],
             local_weight_updated:Iterable[Parameter],
             d_p_list: List[Tensor],
             momentum_buffer_list: List[Optional[Tensor]],
             weight_decay: float,
             momentum: float,
             lr: float,
             dampening: float,
             nesterov: bool,
             prox:float) -> NoReturn:
    """
    """
    for i, (param, localweight) in enumerate(zip(params, local_weight_updated)):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)  # L2 regularization

        if prox != 0:
            d_p = d_p.add(param-localweight, alpha=prox)  # proximal regularization

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)
