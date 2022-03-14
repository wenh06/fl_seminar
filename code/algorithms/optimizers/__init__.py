"""
"""

from typing import Iterable, Union, Any

from torch.nn.parameter import Parameter
from torch.optim import Optimizer
import torch.optim as opt
import torch_optimizer as topt

from .base import ProxSGD
from .fedpd import FedPD_SGD, FedPD_VR, PSGD, PSVRG
from .pfedme import pFedMeOptimizer as pFedMe
from .fedprox import FedProxOptimizer as FedProx
from .feddr import FedDROptimizer as FedDR


__all__ = [
    "FedPD_SGD", "FedPD_VR", "PSGD", "PSVRG", "pFedMe",
    "FedProx", "FedDR",
    "get_optimizer",
]


def get_optimizer(optimizer_name:str, params:Iterable[Union[dict,Parameter]], config:Any) -> Optimizer:
    """ get optimizer by name

    Usage examples
    --------------
    ```python
    import torch

    model = torch.nn.Linear(10, 1)
    optimizer = get_optimizer("SGD", model.parameters(), {"lr": 1e-2})  # PyTorch built-in
    optimizer = get_optimizer("yogi", model.parameters(), {"lr": 1e-2})  # from pytorch_optimizer
    optimizer = get_optimizer("FedPD_SGD", model.parameters(), {"lr": 1e-2})  # federated
    """
    try:
        optimizer = eval(f"opt.{optimizer_name}(params, **config)")
        print(f"PyTorch built-in optimizer {optimizer_name} is used.")
        return optimizer
    except:
        try:
            optimizer = topt.get(optimizer_name)(params, **config)
            print(f"Optimizer {optimizer_name} from torch_optimizer is used.")
            return optimizer
        except:
            pass

    if optimizer_name == "FedPD_SGD":
        return FedPD_SGD(params, mu=config.mu, lamda=config.lamda, lr=config.lr)
    elif optimizer_name == "FedPD_VR":
        return FedPD_VR(params, mu=config.mu, freq_1=config.freq_1, freq_2=config.freq_2, lr=config.lr)
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
