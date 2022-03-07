"""
pFedMe re-implemented in the experiment framework
"""

from copy import deepcopy
import warnings
from typing import List, NoReturn, Dict

import torch
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

from .nodes import Server, Client, SeverConfig, ClientConfig
from .optimizer import get_optimizer


__all__ = [
    "pFedMeServer", "pFedMeClient",
    "pFedMeServerConfig", "pFedMeClientConfig",
]


class pFedMeServerConfig(SeverConfig):
    """
    """
    __name__ = "pFedMeServerConfig"

    def __init__(self,
                 num_iters:int,
                 num_clients:int,
                 clients_sample_ratio:float,
                 beta:float=1.0,) -> NoReturn:
        """
        """
        super().__init__(
            "pFedMe",
            num_iters, num_clients, clients_sample_ratio,
            beta=beta,
        )


class pFedMeClientConfig(ClientConfig):
    """
    """
    __name__ = "pFedMeClientConfig"

    def __init__(self,
                 batch_size:int,
                 num_epochs:int,
                 lr:float=0.09,
                 num_steps:int=30,
                 lamda:float=15.0,
                 mu:float=0.001,) -> NoReturn:
        """
        """
        super().__init__(
            "pFedMe", "pFedMe",
            batch_size, num_epochs, lr,
            num_steps=num_steps, lamda=lamda, mu=mu,
        )


class pFedMeServer(Server):
    """
    """
    __name__ = "pFedMeServer"

    @property
    def client_cls(self) -> "Client":
        return pFedMeClient

    @property
    def required_config_fields(self) -> List[str]:
        """
        """
        return ["beta",]
    
    def communicate(self, target:"pFedMeClient") -> NoReturn:
        """
        """
        target._received_messages = {"parameters": deepcopy(list(self.model.parameters()))}
        self._num_communications += 1

    def update(self) -> NoReturn:
        """
        """
        if len(self._received_messages) == 0:
            warnings.warn("No message received from the clients, unable to update server model")
            return
        # store previous parameters
        previous_param = deepcopy(list(self.model.parameters()))
        for p in previous_param:
            p = p.to(self.device)

        # sum of received parameters, with self.model.parameters() as its container
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_samples = sum([m["train_samples"] for m in self._received_messages])
        for m in self._received_messages:
            self.add_parameters(m["parameters"], m["train_samples"] / total_samples)

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.config.beta) * pre_param.data.detach().clone() + self.config.beta * param.data

        # clear received messages
        del pre_param
        self._received_messages = []


class pFedMeClient(Client):
    """
    """
    __name__ = "pFedMeClient"

    @property
    def required_config_fields(self) -> List[str]:
        """
        """
        return ["num_steps", "lamda",]

    def communicate(self, target:"pFedMeServer") -> NoReturn:
        """
        """
        target._received_messages.append(
            {
                "parameters": deepcopy(list(self.model.parameters())),
                "train_samples": self.config.num_epochs * self.config.num_steps * self.config.batch_size,
            }
        )
        self._received_messages = {}

    def update(self) -> NoReturn:
        """
        """
        try:
            self._client_parameters = deepcopy(self._received_messages["parameters"])
        except KeyError:
            warnings.warn("No parameters received from server")
            warnings.warn("Using current model parameters as initial parameters")
            self._client_parameters = deepcopy(list(self.model.parameters()))
        self._client_parameters = [p.to(self.device) for p in self._client_parameters]
        self.train()

    def train(self) -> NoReturn:
        """
        """
        self.model.train()
        with tqdm(range(self.config.num_epochs), total=self.config.num_epochs) as pbar:
            for epoch in pbar:  # local update
                self.model.train()
                X, y = self.sample_data()
                X, y = X.to(self.device), y.to(self.device)
                # personalized steps
                for i in range(self.config.num_steps):
                    self.optimizer.zero_grad()
                    output = self.model(X)
                    loss = self.criterion(output, y)
                    loss.backward()
                    self.optimizer.step(self._client_parameters)

                # update local weight after finding aproximate theta
                # pFedMe paper Algorithm 1 line 8
                for up, cp in zip(self.model.parameters(), self._client_parameters):
                    cp.data.add_(cp.data - up.data, alpha=-self.config.lamda * self.config.lr)

                # update local model
                # the init parameters (theta in pFedMe paper Algorithm 1 line  7)
                # are set to be `self._client_parameters`
                self.set_parameters(self._client_parameters)

    def evaluate(self) -> Dict[str, float]:
        """
        """
        self.model.eval()
        _metrics = []
        for X, y in self.val_loader:
            logits = self.model(X)
            _metrics.append(self.dataset.evaluate(logits, y))
        metrics = {"num_examples": sum([m["num_examples"] for m in _metrics]),}
        for k in _metrics[0]:
            if k != "num_examples":  # average over all metrics
                metrics[k] = sum([m[k] * m["num_examples"] for m in _metrics]) / metrics["num_examples"]
        return metrics
