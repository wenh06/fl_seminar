"""
"""

from copy import deepcopy
import warnings
from typing import List, NoReturn, Dict, Sequence

import torch
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

from nodes import Server, Client, ServerConfig, ClientConfig, ClientMessage
from ..optimizers import get_optimizer


__all__ = [
    "FedOptServer", "FedOptClient",
    "FedOptServerConfig", "FedOptClientConfig",
]


class FedOptServerConfig(ServerConfig):
    """
    """
    __name__ = "FedOptServerConfig"

    def __init__(self,
                 num_iters:int,
                 num_clients:int,
                 clients_sample_ratio:float,
                 optimizer:str="Adam",
                 lr:float=1e-3,
                 betas:Sequence[float]=(0.9, 0.999),
                 tau:float=1e-5,) -> NoReturn:
        """
        tau: controls the degree of adaptivity of the algorithm
        """
        assert optimizer.lower() in [
            "avg", "adagrad", "yogi", "adam",
        ]
        super().__init__(
            "FedOpt",
            num_iters, num_clients, clients_sample_ratio,
            optimizer=optimizer, lr=lr, betas=betas, tau=tau,
        )


class FedOptClientConfig(ClientConfig):
    """
    """
    __name__ = "FedOptClientConfig"

    def __init__(self,
                 batch_size:int,
                 num_epochs:int,
                 lr:float=1e-3,
                 optimizer:str="SGD",
                 **kwargs,) -> NoReturn:
        """
        """
        super().__init__(
            "FedOpt", optimizer,
            batch_size, num_epochs, lr,
            **kwargs,
        )


class FedOptServer(Server):
    """
    """
    __name__ = "FedOptServer"

    def _post_init(self) -> NoReturn:
        """
        """
        super()._post_init()
        self.delta_parameters = deepcopy(list(self.model.parameters()))
        for p in self.delta_parameters:
            p.data.zero_()
        if self.config.optimizer.lower() != "avg":
            self.v_parameters = deepcopy(self.delta_parameters)
            for p in self.v_parameters:
                p.data.random_(1, 100).mul_(self.config.tau**2)
        else:
            self.config.lr = 1
            self.v_parameters = [
                torch.Tensor([1-self.config.tau]).to(self.model.dtype).to(self.device) for _ in self.delta_parameters
            ]

    @property
    def client_cls(self) -> "Client":
        return FedOptClient

    @property
    def required_config_fields(self) -> List[str]:
        """
        """
        return ["optimizer", "lr", "betas", "tau",]
    
    def communicate(self, target:"FedOptClient") -> NoReturn:
        """
        """
        target._received_messages = {"parameters": deepcopy(list(self.model.parameters()))}

    def update(self) -> NoReturn:
        """
        """
        # update delta_parameters, FedOpt paper Algorithm 2, line 10
        for idx, param in enumerate(self.delta_parameters):
            param.data.mul_(self.config.betas[0])
            for m in self._received_messages:
                param.data.add_(
                    m["parameters"][idx].data.detach().clone().to(self.device),
                    alpha=(1 - self.config.betas[0]) * m["train_samples"] / total_samples
                )
        # update v_parameters, FedOpt paper Algorithm 2, line 11-13
        optimizer = self.config.optimizer.lower()
        if optimizer == "avg":
            self.update_avg()
        elif optimizer == "adagrad":
            self.update_adagrad()
        elif optimizer == "yogi":
            self.update_yogi()
        elif optimizer == "adam":
            self.update_adam()
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        # update model parameters, FedOpt paper Algorithm 2, line 14
        for sp, dp, vp in zip(self.model.parameters(), self.delta_parameters, self.v_parameters):
            sp.data.addcdiv_(
                dp.data, vp.sqrt()+self.config.tau,
                value=-self.config.lr,
            )

    def update_avg(self) -> NoReturn:
        """
        """
        pass

    def update_adagrad() -> NoReturn:
        """
        """
        for vp, dp in zip(self.v_parameters, self.delta_parameters):
            vp.data.add_(dp.data.pow(2))

    def update_yogi() -> NoReturn:
        """
        """
        for vp, dp in zip(self.v_parameters, self.delta_parameters):
            vp.data.addcmul_(
                dp.data.pow(2), (vp.data-dp.data.pow(2)).sign(),
                value=-(1-self.config.betas[1])
            )

    def update_adam() -> NoReturn:
        """
        """
        for vp, dp in zip(self.v_parameters, self.delta_parameters):
            vp.data.mul_(self.config.betas[1]).add_(dp.data.pow(2), alpha=1-self.config.betas[1])

class FedOptClient(Client):
    """
    """
    __name__ = "FedOptClient"

    @property
    def required_config_fields(self) -> List[str]:
        """
        """
        return ["optimizer",]

    def communicate(self, target:"FedOptServer") -> NoReturn:
        """
        """
        delta_parameters = deepcopy(list(self.model.parameters()))
        for dp, rp in zip(delta_parameters, self._received_messages["parameters"]):
            dp.data.add_(rp.data, alpha=-1)
        target._received_messages.append(ClientMessage(
            **{
                "client_id": self.client_id,
                "delta_parameters": delta_parameters,
                "train_samples": len(self.train_loader.dataset),
                "metrics": self._metrics,
            }
        ))

    def update(self) -> NoReturn:
        """
        """
        try:
            self.set_parameters(self._received_messages["parameters"])
        except KeyError:
            warnings.warn("No parameters received from server")
            warnings.warn("Using current model parameters as initial parameters")
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
                    self.optimizer.step()
