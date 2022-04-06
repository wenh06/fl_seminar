"""
pFedMe re-implemented in the new framework
"""

from copy import deepcopy
import warnings
from typing import List, NoReturn, Any

import torch  # noqa: F401

try:
    from tqdm.auto import tqdm  # noqa: F401
except ImportError:
    from tqdm import tqdm  # noqa: F401

from nodes import (
    Server,
    Client,
    ServerConfig,
    ClientConfig,
    ClientMessage,
)  # noqa: F401


__all__ = [
    "pFedMeServer",
    "pFedMeClient",
    "pFedMeServerConfig",
    "pFedMeClientConfig",
]


class pFedMeServerConfig(ServerConfig):
    """ """

    __name__ = "pFedMeServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        beta: float = 1.0,
        **kwargs: Any,
    ) -> NoReturn:
        """ """
        super().__init__(
            "pFedMe", num_iters, num_clients, clients_sample_ratio, beta=beta, **kwargs
        )


class pFedMeClientConfig(ClientConfig):
    """
    References
    ----------
    1. https://github.com/CharlieDinh/pFedMe/blob/master/FLAlgorithms/users/userpFedMe.py

    Note:
    1. `lr` is the `personal_learning_rate` in the original implementation
    2. `eta` is the `learning_rate` in the original implementation
    3. `mu` is the momentum factor in the original implemented optimzer
    """

    __name__ = "pFedMeClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 5e-3,
        num_steps: int = 30,
        lamda: float = 15.0,
        eta: float = 1e-3,
        mu: float = 1e-3,
        **kwargs: Any,
    ) -> NoReturn:
        """ """
        super().__init__(
            "pFedMe",
            "pFedMe",
            batch_size,
            num_epochs,
            lr,
            num_steps=num_steps,
            lamda=lamda,
            eta=eta,
            mu=mu,
            **kwargs,
        )


class pFedMeServer(Server):
    """ """

    __name__ = "pFedMeServer"

    @property
    def client_cls(self) -> "Client":
        return pFedMeClient

    @property
    def required_config_fields(self) -> List[str]:
        """ """
        return [
            "beta",
        ]

    def communicate(self, target: "pFedMeClient") -> NoReturn:
        """ """
        target._received_messages = {
            "parameters": deepcopy(list(self.model.parameters()))
        }

    def update(self) -> NoReturn:
        """ """
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
            param.data = (
                1 - self.config.beta
            ) * pre_param.data.detach().clone() + self.config.beta * param.data

        # clear received messages
        del pre_param


class pFedMeClient(Client):
    """ """

    __name__ = "pFedMeClient"

    @property
    def required_config_fields(self) -> List[str]:
        """ """
        return [
            "num_steps",
            "lamda",
            "eta",
            "mu",
        ]

    def communicate(self, target: "pFedMeServer") -> NoReturn:
        """ """
        target._received_messages.append(
            ClientMessage(
                **{
                    "client_id": self.client_id,
                    "parameters": deepcopy(list(self.model.parameters())),
                    "train_samples": self.config.num_epochs * self.config.batch_size,
                    "metrics": self._metrics,
                }
            )
        )

    def update(self) -> NoReturn:
        """ """
        # copy the parameters from the server
        # pFedMe paper Algorithm 1 line 5
        try:
            self._cached_parameters = deepcopy(self._received_messages["parameters"])
        except KeyError:
            warnings.warn("No parameters received from server")
            warnings.warn("Using current model parameters as initial parameters")
            self._cached_parameters = deepcopy(list(self.model.parameters()))
        except Exception as err:
            raise err
        self._cached_parameters = [p.to(self.device) for p in self._cached_parameters]
        # update the model via prox_sgd
        # pFedMe paper Algorithm 1 line 6 - 8
        self.train()

    def train(self) -> NoReturn:
        """ """
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
                    self.optimizer.step(self._cached_parameters)

                # update local weight after finding aproximate theta
                # pFedMe paper Algorithm 1 line 8
                for mp, cp in zip(self.model.parameters(), self._cached_parameters):
                    # print(mp.data.isnan().any(), cp.data.isnan().any())
                    cp.data.add_(
                        cp.data.clone() - mp.data.clone(),
                        alpha=-self.config.lamda * self.config.eta,
                    )

                # update local model
                # the init parameters (theta in pFedMe paper Algorithm 1 line  7) for the next iteration
                # are set to be `self._cached_parameters`
                self.set_parameters(self._cached_parameters)
