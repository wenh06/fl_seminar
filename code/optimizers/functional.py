"""
"""

from typing import List, Iterable, Optional

import torch
from torch import Tensor
from torch.nn import Parameter


__all__ = [
    "prox_sgd",
    "al_sgd",
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
) -> None:
    r"""
    The function that executes the proximal SGD:
        .. math::
            \DeclareMathOperator*{\argmin}{arg\,min}
            \operatorname{prox}_{\rho f}(v) = \argmin_x \{f(x) + \dfrac{\rho}{2} \lVert x-v \rVert_2^2\}

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
        the list of momentum buffers
    weight_decay: float,
        weight decay (L2 penalty)
    momentum: float,
        momentum factor
    lr: float,
        the learning rate
    dampening: float,
        dampening for momentum
    nesterov: bool,
        if True, enables Nesterov momentum
    prox: float,
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


def al_sgd(
    params: List[Tensor],
    local_weights: List[Tensor],
    dual_weights: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    mu: float,
) -> None:
    r"""
    The function that executes the augmented Lagrangian SGD:
        .. math::
            \DeclareMathOperator*{\argmin}{arg\,min}
            \argmin_x \mathcal{L}_{\mu}(x, x_0, \lambda) = \argmin_x \{f(x) + \langle \lambda, x-x_0 \rangle + \dfrac{1}{2\mu} \lVert x-x_0 \rVert_2^2\}

    Parameters
    ----------
    params: list of dict or Parameter,
        the parameters to optimize
    local_weights: iterable of Parameter,
        the (init) local weights,
        i.e. the term `x_0` in
        .. math::
            \mathcal{L}_{\mu}(x, x_0, \lambda)
    dual_weights: iterable of Parameter,
        the weights of dual variables,
        i.e. the term `\lambda` in
        .. math::
            \mathcal{L}_{\mu}(x, x_0, \lambda)
    d_p_list: list of Tensor,
        the list of gradients of the parameters
    momentum_buffer_list: list of Tensor or list of None,
        the list of momentum buffers,
        works only if `momentum` > 0
    gradient_variance_buffer_list: list of Tensor or list of None,
        the list of gradient variance buffers,
        works only is `vr` is True
    weight_decay: float,
        weight decay (L2 penalty)
    momentum: float,
        momentum factor
    lr: float, default 1e-3,
        the learning rate
    dampening: float,
        dampening for momentum
    nesterov: bool,
        if True, enables Nesterov momentum
    mu: float,
        the (penalty) coeff. of the augmented Lagrangian term,
        i.e. the term `\mu` in
        .. math::
            \mathcal{L}_{\mu}(x, x_0, \lambda)

    """
    for idx, (param, lw, dw) in enumerate(zip(params, local_weights, dual_weights)):

        d_p = d_p_list[idx]

        d_p = d_p.add(dw.detach().clone())

        d_p = d_p.add(param - lw.detach().clone(), alpha=1 / mu)

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)  # L2 regularization

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
