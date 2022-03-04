"""
"""

from typing import Union

import numpy as np
import torch
from torch import Tensor
from torch import nn


__all__ = [
    "compute_module_size",
    "SizeMixin",
    "top_n_accuracy",
]


def compute_module_size(module:nn.Module, human:bool=False, dtype:str="float32") -> Union[int, str]:
    """ finished, checked,

    compute the size (number of parameters) of a module

    Parameters
    ----------
    module: Module,
        a torch Module
    human: bool, default False,
        return size in a way that is easy to read by a human,
        by appending a suffix corresponding to the unit (K, M, G, T, P)
    dtype: str, default "float32",
        data type of the module parameters, one of "float16", "float32", "float64"
    
    Returns
    -------
    n_params: int,
        size (number of parameters) of this torch module
    """
    module_parameters = filter(lambda p: p.requires_grad, module.parameters())
    n_params = sum([np.prod(p.size()) for p in module_parameters])
    if human:
        n_params = n_params * {"float16":2, "float32":4, "float64":8}[dtype.lower()] / 1024
        div_count = 0
        while n_params >= 1024:
            n_params /= 1024
            div_count += 1
        # cvt_dict = {0:"K", 1:"M", 2:"G", 3:"T", 4:"P"}
        cvt_dict = {c:u for c,u in enumerate(list("KMGTP"))}
        n_params = f"""{n_params:.1f}{cvt_dict[div_count]}"""
    return n_params


class SizeMixin(object):
    """ finished, checked,

    mixin class for size related methods
    """
    
    @property
    def module_size(self) -> int:
        return compute_module_size(self)

    @property
    def module_size_(self) -> str:
        try:
            dtype = str(next(self.parameters()).dtype).replace("torch.", "")
        except StopIteration:
            dtype = "float32"  # can be set arbitrarily among all the supported types
        return compute_module_size(
            self, human=True, dtype=dtype,
        )


def top_n_accuracy(preds:Tensor, labels:Tensor, n:int=1) -> float:
    """
    """
    _, indices = torch.topk(preds, n, dim=1)  # of shape (N, n)
    correct = torch.sum(indices == labels.repeat(n, 1).T)
    return correct.item() / preds.shape[0]
