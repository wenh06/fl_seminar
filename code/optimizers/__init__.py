"""
"""

import inspect
from typing import Iterable, Union, Any

from torch.nn.parameter import Parameter
from torch.optim import Optimizer
import torch.optim as opt  # noqa: F401
import torch_optimizer as topt  # noqa: F401
from easydict import EasyDict as ED

from misc import isclass
from .base import ProxSGD  # noqa: F401
from .fedpd import FedPD_SGD, FedPD_VR, PSGD, PSVRG  # noqa: F401
from .pfedme import pFedMeOptimizer as pFedMe  # noqa: F401
from .fedprox import FedProxOptimizer as FedProx  # noqa: F401
from .feddr import FedDROptimizer as FedDR  # noqa: F401


__all__ = [
    "ProxSGD",
    "FedPD_SGD",
    "FedPD_VR",
    "PSGD",
    "PSVRG",
    "pFedMe",
    "FedProx",
    "FedDR",
    "get_optimizer",
]


def get_optimizer(
    optimizer_name: Union[str, type],
    params: Iterable[Union[dict, Parameter]],
    config: Any,
) -> Optimizer:
    """get optimizer by name

    Usage examples
    --------------
    ```python
    import torch

    model = torch.nn.Linear(10, 1)
    optimizer = get_optimizer("SGD", model.parameters(), {"lr": 1e-2})  # PyTorch built-in
    optimizer = get_optimizer("yogi", model.parameters(), {"lr": 1e-2})  # from pytorch_optimizer
    optimizer = get_optimizer("FedPD_SGD", model.parameters(), {"lr": 1e-2})  # federated
    """
    if isclass(optimizer_name) and issubclass(optimizer_name, Optimizer):
        # print(f"{optimizer_name.__name__} is an Optimizer class, used directly")
        return optimizer_name(params, **_get_args(optimizer_name, config))
    try:
        _config = _get_args(eval(f"opt.{optimizer_name}"), config)
        optimizer = eval(f"opt.{optimizer_name}(params, **_config)")
        # print(f"PyTorch built-in optimizer {optimizer_name} is used.")
        return optimizer
    except Exception:
        try:
            optimizer = topt.get(optimizer_name)(
                params, **_get_args(topt.get(optimizer_name), config)
            )
            # print(f"Optimizer {optimizer_name} from torch_optimizer is used.")
            return optimizer
        except Exception:
            pass

    if isinstance(config, dict):
        config = ED(config)
    if optimizer_name == "FedPD_SGD":
        return FedPD_SGD(params, mu=config.mu, lamda=config.lamda, lr=config.lr)
    elif optimizer_name == "FedPD_VR":
        return FedPD_VR(
            params,
            mu=config.mu,
            freq_1=config.freq_1,
            freq_2=config.freq_2,
            lr=config.lr,
        )
    elif optimizer_name == "PSGD":
        pass
    elif optimizer_name == "PSVRG":
        pass
    elif optimizer_name == "pFedMe":
        return pFedMe(params, lr=config.lr, lamda=config.lamda, mu=config.mu)
    elif optimizer_name == "FedProx":
        return FedProx(params, lr=config.lr, mu=config.mu)
    elif optimizer_name == "FedDR":
        return FedDR(params, lr=config.lr, eta=config.eta)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")


def _get_args(cls: type, config: Any) -> ED:
    """
    used to filter out the items in config that are not arguments of the class
    """
    if isinstance(config, dict):
        config = ED(config)
    args = [
        k
        for k in inspect.getfullargspec(cls.__init__).args
        if k
        not in [
            "self",
            "params",
        ]
    ]
    kwargs = ED()
    for k in args:
        try:
            kwargs[k] = eval(f"config.{k}")
        except Exception:
            pass
    return kwargs
