"""
"""

from .fedpd import FedPD_SGD, FedPD_VR, PSGD, PSVRG
from .pfedme import pFedMeOptimizer as pFedMe


__all__ = [
    "FedPD_SGD", "FedPD_VR", "PSGD", "PSVRG", "pFedMe",
    "get_optimizer",
]


def get_optimizer(optimizer_name:str, params:Iterable[Union[dict,Parameter]], config:Any) -> Optimizer:
    """
    """
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
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")
