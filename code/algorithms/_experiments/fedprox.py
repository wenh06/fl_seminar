"""
FedProx re-implemented in the experiment framework
"""

from copy import deepcopy
import warnings
from typing import List, NoReturn, Dict

import torch
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

from nodes import Server, Client, ServerConfig, ClientConfig, ClientMessage
from ..optimizers import get_optimizer


__all__ = [
    "FedProxServer", "FedProxClient",
    "FedProxServerConfig", "FedProxClientConfig",
]


class FedProxServerConfig(ServerConfig):
    """
    """
    __name__ = "FedProxServerConfig"

    def __init__(self,
                 num_iters:int,
                 num_clients:int,
                 clients_sample_ratio:float,) -> NoReturn:
        """
        """
        super().__init__(
            "FedProx",
            num_iters, num_clients, clients_sample_ratio,
        )


class FedProxClientConfig(ClientConfig):
    """
    """
    __name__ = "FedProxClientConfig"

    def __init__(self,
                 batch_size:int,
                 num_epochs:int,
                 lr:float=1e-3,
                 mu:float=0.01,) -> NoReturn:
        """
        """
        super().__init__(
            "FedProx", "FedProx",
            batch_size, num_epochs,
            lr, mu=mu,
        )


class FedProxServer(Server):
    """
    """
    __name__ = "FedProxServer"

    @property
    def client_cls(self) -> "Client":
        return FedProxClient

    @property
    def required_config_fields(self) -> List[str]:
        """
        """
        return []
    
    def communicate(self, target:"FedProxClient") -> NoReturn:
        """
        """
        target._received_messages = {"parameters": deepcopy(list(self.model.parameters()))}

    def update(self) -> NoReturn:
        """
        """
        # sum of received parameters, with self.model.parameters() as its container
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_samples = sum([m["train_samples"] for m in self._received_messages])
        for m in self._received_messages:
            self.add_parameters(m["parameters"], m["train_samples"] / total_samples)


class FedProxClient(Client):
    """
    """
    __name__ = "FedProxClient"

    @property
    def required_config_fields(self) -> List[str]:
        """
        """
        return ["mu",]

    def communicate(self, target:"FedProxServer") -> NoReturn:
        """
        """
        target._received_messages.append(ClientMessage(
            **{
                "client_id": self.client_id,
                "parameters": deepcopy(list(self.model.parameters())),
                "train_samples": len(self.train_loader.dataset),
                "metrics": self._metrics,
            }
        ))

    def update(self) -> NoReturn:
        """
        """
        try:
            self._client_parameters = deepcopy(self._received_messages["parameters"])
        except KeyError:
            warnings.warn("No parameters received from server")
            warnings.warn("Using current model parameters as initial parameters")
            self._client_parameters = deepcopy(list(self.model.parameters()))
        except Exception as err:
            raise err
        self._client_parameters = [p.to(self.device) for p in self._client_parameters]
        self.train()

    def train(self) -> NoReturn:
        """
        """
        self.model.train()
        with tqdm(range(self.config.num_epochs), total=self.config.num_epochs) as pbar:
            for epoch in pbar:  # local update
                self.model.train()
                for X, y in self.train_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(X)
                    loss = self.criterion(output, y)
                    loss.backward()
                    self.optimizer.step(self._client_parameters)
