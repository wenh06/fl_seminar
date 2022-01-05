"""
"""

from typing import Iterable, Union, NoReturn, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer



class FedPD_VR(Optimizer):
    """
    """
    def __init__(self,
                 params:Iterable[Union[dict,Tensor]],
                 lr:float=1e-3, mu:float=1.0, freq_1:int=10, freq_2:int=10,) -> NoReturn:
        """

        Parameters
        ----------
        to write
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, freq_1=freq_1+1, mu=mu, freq_2=freq_2)
        self.counter_in = 0
        self.counter_out = 0
        self.flag = False
        super().__init__(params, defaults)

    def step(self, closure:Optional[callable]=None) -> Optional[Tensor]:
        """Performs a single optimization step.

        Parameters
        ----------
        closure: callable, optional,
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        loss: Tensor, optional,
            The loss after the step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            freq_1 = group["freq_1"]
            freq_2 = group["freq_2"]
            mu = group["mu"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]

                if not (self.flag):
                    param_state["x_0"] = torch.zeros_like(p.data)
                    param_state["g_ex"] = torch.zeros_like(p.data)
                    param_state["lambda"] = torch.zeros_like(p.data)
                
                x_0 = param_state["x_0"]
                g_ex = param_state["g_ex"]
                lamb = param_state["lambda"]

                if (self.counter_in == 0):
                    if not (self.flag): # first iteration, initialize
                        x_0.copy_(p.data) # x_0 = x_i
                    else: # after the first iteration
                        temp = p.data.clone().detach()
                        p.data.copy_(x_0)
                        x_0.copy_(temp)

                    if (self.counter_out == 0):
                        g_ex.fill_(0) # g_ex = 0
                    
                g_ex.add_(d_p) # g_ex = g_ex + (h-h')

                if (self.counter_in > 0): # first inner loop, only switch x_0 and x_i
                    p.data.add_(-group["lr"]*mu, p.data-x_0)
                    p.data.add_(-group["lr"], g_ex + lamb)

                if (self.counter_in+1 == freq_1): # last inner loop, perform update on lambda and x_0
                    lamb.add_(mu, p.data - x_0)
                    x_0.copy_(p.data)
                    p.data.add_(1./mu, lamb)
                
        self.flag = True
        self.counter_in += 1    
        if self.counter_in == freq_1:
            self.counter_in = 0
            self.counter_out += 1
            if self.counter_out == freq_2:
                self.counter_out = 0

        return loss

class FedPD_SGD(Optimizer):
    """
    """

    def __init__(self,
                 params:Iterable[Union[dict,Tensor]],
                 lr:float=1e-3, mu:float=1.0, freq:int=10,) -> NoReturn:
        """

        Parameters
        ----------
        to write
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, freq=freq+1, mu=mu)
        self.counter_in = 0
        self.flag = False
        super().__init__(params, defaults)

    def step(self, closure:Optional[callable]=None) -> Optional[Tensor]:
        """Performs a single optimization step.

        Parameters
        ----------
        closure: callable, optional,
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        loss: Tensor, optional,
            The loss after the step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            freq = group["freq"]
            mu = group["mu"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]

                if not (self.flag):
                    # print("inner_init")
                    param_state["x_0"] = torch.zeros_like(p.data)
                    param_state["lambda"] = torch.zeros_like(p.data)
                
                x_0 = param_state["x_0"]
                lamb = param_state["lambda"]

                if (self.counter_in == 0):
                    if not (self.flag): # first iteration, initialize
                        x_0.copy_(p.data) # x_0 = x_i
                    else: # after the first iteration
                        temp = p.data.clone().detach()
                        p.data.copy_(x_0)
                        x_0.copy_(temp)

                if (self.counter_in > 0): # first inner loop, only switch x_0 and x_i
                    p.data.add_(-group["lr"]*mu, p.data-x_0)
                    p.data.add_(-group["lr"], d_p+ lamb)

                if (self.counter_in+1 == freq): # last inner loop, perform update on lambda and x_0
                    lamb.add_(mu, p.data - x_0)
                    x_0.copy_(p.data)
                    p.data.add_(1./mu, lamb)
                
        self.flag = True
        self.counter_in += 1    
        if self.counter_in == freq:
            self.counter_in = 0

        return loss



# -----------------------------
# the following for comparison


class PSVRG(Optimizer):
    """
    might mistake for FSVRG, to check
    """

    def __init__(self,
                 params:Iterable[Union[dict,Tensor]],
                 lr:float=1e-3, mu:float=1.0, freq:int=10,) -> NoReturn:
        """

        Parameters
        ----------
        to write
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, freq=freq, mu=mu)
        self.counter = 0
        self.flag = False
        super().__init__(params, defaults)

    def step(self, closure:Optional[callable]=None) -> Optional[Tensor]:
        """Performs a single optimization step.

        Parameters
        ----------
        closure: callable, optional,
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        loss: Tensor, optional,
            The loss after the step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            freq = group["freq"]
            mu = group["mu"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]

                if not (self.flag):
                    param_state["x_0"] = torch.zeros_like(p.data)
                    param_state["g_ex"] = torch.zeros_like(p.data)
                
                x_0 = param_state["x_0"]
                g_ex = param_state["g_ex"]

                if (self.counter ==0):
                    x_0.copy_(p.data)
                    g_ex.fill_(0)
                
                g_ex.add_(d_p)

                p.data.add_(-group["lr"]*mu, p.data-x_0)
                p.data.add_(-group["lr"], g_ex)

        self.flag = True
        self.counter += 1    
        if self.counter == freq:
            self.counter = 0
        return loss


class PSGD(Optimizer):
    """
    might mistake for PR-SGD, to check
    """
    
    def __init__(self,
                 params:Iterable[Union[dict,Tensor]],
                 lr:float=1e-3, mu:float=1.0, freq:int=2,) -> NoReturn:
        """

        Parameters
        ----------
        to write
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, mu= mu, freq= freq)
        self.counter = 0
        self.flag = False
        super().__init__(params, defaults)

    def step(self, closure:Optional[callable]=None) -> Optional[Tensor]:
        """Performs a single optimization step.

        Parameters
        ----------
        closure: callable, optional,
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        loss: Tensor, optional,
            The loss after the step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            mu = group["mu"]
            freq = group["freq"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]

                if not (self.flag):
                    param_state["x_0"] = torch.zeros_like(p.data)
                
                x_0 = param_state["x_0"]

                if self.counter == 0:
                    x_0.copy_(p.data)

                p.data.add_(-group["lr"]*mu, p.data-x_0)
                p.data.add_(-group["lr"], d_p)

        self.flag = True
        self.counter += 1    
        if self.counter == freq:
            self.counter = 0
        return loss
