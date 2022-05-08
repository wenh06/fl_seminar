"""
"""

import warnings
from typing import Iterable, Union, NoReturn, Optional

import torch  # noqa: F401
from torch import Tensor
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer

from . import functional as F


__all__ = [
    "ProxSGD",
    "AL_SGD",
]


class ProxSGD(Optimizer):
    r"""
    Proximal Stochastic Gradient Descent.
    Using SGD to solve the proximal problem:
        .. math::
            \DeclareMathOperator*{\argmin}{arg\,min}
            \operatorname{prox}_{\rho f}(v) = \argmin_x \{f(x) + \dfrac{\rho}{2} \lVert x-v \rVert_2^2\}
    when it does not have a closed form solution.

    """

    __name__ = "ProxSGD"

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 1e-3,
        momentum=1e-3,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        prox: float = 0.1,
    ) -> NoReturn:
        r"""

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
        weight_decay: float, default 0,
            weight decay (L2 penalty)
        nesterov: bool, default False,
            if True, enables Nesterov momentum
        prox: float, default 0.1,
            the (penalty) coeff. of the proximal term,
            i.e. the term `\rho` in
            .. math::
                \argmin_x \{f(x) + \dfrac{\rho}{2} \lVert x-v \rVert_2^2\}

        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        if prox < 0.0:
            raise ValueError(f"Invalid prox value: {prox}")
        if prox * lr >= 1:
            warnings.warn(
                f"prox * lr = {prox * lr:.3f} >= 1 with prox = {prox}, lr = {lr}, you may encounter gradient exploding",
            )
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            prox=prox,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state: dict) -> NoReturn:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(
        self,
        local_weight_updated: Iterable[Parameter],
        closure: Optional[callable] = None,
    ) -> Optional[Tensor]:
        r"""

        Parameters
        ----------
        local_weight_updated: iterable of Parameter,
            the local weights updated by the local optimizer,
            or of the previous iteration,
            i.e. the term `v` in
            .. math::
                \argmin_x \{f(x) + \dfrac{\rho}{2} \lVert x-v \rVert_2^2\}
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

            F.prox_sgd(
                params_with_grad,
                local_weight_updated,
                d_p_list,
                momentum_buffer_list,
                weight_decay,
                momentum,
                lr,
                dampening,
                nesterov,
                prox,
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss


class AL_SGD(Optimizer):
    r"""

    Augmented Lagrangian Stochastic Gradient Descent.
    Using SGD to solve the augmented Lagrangian problem:
        .. math::
            \DeclareMathOperator*{\argmin}{arg\,min}
            \argmin_x \mathcal{L}_{\rho}(x, x_0, \lambda) = \argmin_x \{f(x) + \langle \lambda, x-x_0 \rangle + \dfrac{\rho}{2} \lVert x-x_0 \rVert_2^2\}
    when it does not have a closed form solution.

    """

    __name__ = "AL_SGD"

    def __init__(
        self,
        params: Iterable[Union[dict, Parameter]],
        lr: float = 1e-3,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        rho: float = 0.1,
    ) -> NoReturn:
        r"""

        Parameters
        ----------
        params: iterable of dict or Parameter,
            the parameters to optimize
        lr: float, default 1e-3,
            the learning rate
        momentum: float, default 0,
            momentum factor
        dampening: float, default 0,
            dampening for momentum
        weight_decay: float, default 0,
            weight decay (L2 penalty)
        nesterov: bool, default False,
            if True, enables Nesterov momentum
        rho: float, default 0.1,
            the (penalty) coeff. of the augmented Lagrangian term,
            i.e. the term `\rho` in
            .. math::
                \mathcal{L}_{\rho}(x, x_0, \lambda)

        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        if rho < 0.0:
            raise ValueError(f"Invalid rho value: {rho}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            rho=rho,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state: dict) -> NoReturn:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(
        self,
        local_weight: Iterable[Parameter],
        dual_weight: Iterable[Parameter],
        closure: Optional[callable] = None,
    ) -> Optional[Tensor]:
        r"""

        Parameters
        ----------
        local_weight: iterable of Parameter,
            the (init) local weights,
            i.e. the term `x_0` in
            .. math::
                \mathcal{L}(x, x_0, \lambda)
        dual_weight: iterable of Parameter,
            the weights of dual variables,
            i.e. the term `\lambda` in
            .. math::
                \mathcal{L}(x, x_0, \lambda)
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
            rho = group["rho"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

            F.al_sgd(
                params_with_grad,
                local_weight,
                dual_weight,
                d_p_list,
                momentum_buffer_list,
                weight_decay,
                momentum,
                lr,
                dampening,
                nesterov,
                rho,
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss
