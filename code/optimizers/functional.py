"""
"""

from typing import List, Iterable, NoReturn, Optional

import torch
from torch import Tensor
from torch.nn import Parameter


__all__ = [
    "prox_sgd",
]


def prox_sgd(
    params: List[Tensor],
    local_weight_updated: Iterable[Parameter],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    prox: float,
) -> NoReturn:
    r"""
    The function that executes the proximal SGD.

    Parameters
    ----------
    params: list of dict or Parameter,
        the parameters to optimize
    local_weight_updated: list of Parameter,
        the local weights updated by the local optimizer,
        or of the previous iteration
    d_p_list: list of Tensor,
        the list of gradients of the parameters
    momentum_buffer_list: list of Tensor or list of None,
        the list of momentum buffers if `nesterov` is True
    weight_decay: float, default 0,
        weight decay (L2 penalty)
    momentum: float, default 1e-3,
        momentum factor
    lr: float, default 1e-3,
        the learning rate
    dampening: float, default 0,
        dampening for momentum
    nesterov: bool, default False,
        if True, enables Nesterov momentum
    prox: float, default 0.1,
        the (penalty) coeff. of the proximal term,
        i.e. the term `\rho` in
        .. math::
            \argmin_x \{f(x) + \dfrac{\rho}{2} \lVert x-v \rVert_2^2\}

    """
    for idx, (param, localweight) in enumerate(zip(params, local_weight_updated)):

        d_p = d_p_list[idx]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)  # L2 regularization

        if prox != 0:
            d_p = d_p.add(
                param - localweight.detach().clone(), alpha=prox
            )  # proximal regularization

        if momentum != 0:
            buf = momentum_buffer_list[idx]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[idx] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)
