"""
"""

from typing import Union, Optional, Dict

import numpy as np
import einops
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

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class CLFMixin(object):
    """
    mixin for classifiers
    """
    __name__ = "CLFMixin"

    def predict_proba(self, input:Tensor, multi_label:bool=False) -> np.ndarray:
        """
        """
        output = self.forward(input)
        if multi_label:
            return torch.sigmoid(output).cpu().detach().numpy()
        return torch.softmax(output, dim=-1).cpu().detach().numpy()

    def predict(self, input:Tensor, thr:Optional[float]=None, class_map:Optional[Dict[int, str]]=None) -> list:
        """
        """
        proba = self.predict_proba(input, multi_label=thr is not None)
        if thr is None:
            output = proba.argmax(axis=-1).tolist()
            if class_map is not None:
                output = [class_map[i] for i in output]
            return output
        output = [[] for _ in range(input.shape[0])]
        indices = np.where(proba > thr)
        if len(indices) > 2:
            raise ValueError(f"multi-label classification is not supported for output of 3 dimensions or more")
        for i, j in zip(*indices):
            output[i].append(j)
        for idx in range(len(output)):
            if len(output[idx]) == 0:
                output[idx] = [proba[idx].argmax()]
        if class_map is not None:
            output = [[class_map[i] for i in l] for l in output]
        return output


def top_n_accuracy(preds:Tensor, labels:Tensor, n:int=1) -> float:
    """
    preds of shape (batch_size, n_classes) or (batch_size, n_classes, d_1, ..., d_n)
    labels of shape (batch_size,) or (batch_size, d_1, ..., d_n)
    """
    assert preds.shape[0] == labels.shape[0]
    batch_size, n_classes, *extra_dims = preds.shape
    _, indices = torch.topk(preds, n, dim=1)  # of shape (batch_size, n) or (batch_size, n, d_1, ..., d_n)
    pattern = " ".join([f"d_{i+1}" for i in range(len(extra_dims))])
    pattern = f"batch_size {pattern} -> batch_size n {pattern}"
    correct = torch.sum(indices == einops.repeat(labels, pattern, n=n))
    acc =  correct.item() / preds.shape[0]
    for d in extra_dims:
        acc = acc / d
    return acc
