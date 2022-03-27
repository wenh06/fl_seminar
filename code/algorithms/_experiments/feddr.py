"""
"""

from copy import deepcopy
import warnings
from typing import List, NoReturn, Dict

import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

from data_processing.fed_dataset import FedDataset
from nodes import Server, Client, ServerConfig, ClientConfig, ClientMessage
from ..optimizers import get_optimizer
from ..regularizers import get_regularizer


__all__ = [
    "FedDRServer",
    "FedDRClient",
    "FedDRServerConfig",
    "FedDRClientConfig",
]


class FedDRServerConfig(ServerConfig):
    """ """

    __name__ = "FedDRServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        eta: float = 1.0,
        reg_type: str = "l1_norm",
    ) -> NoReturn:
        """ """
        super().__init__(
            "FedDR",
            num_iters,
            num_clients,
            clients_sample_ratio,
            eta=eta,
            reg_type=reg_type,
        )


class FedDRClientConfig(ClientConfig):
    """ """

    __name__ = "FedDRClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-3,
        eta: float = 1.0,
        alpha: float = 1.9,
    ) -> NoReturn:
        """ """
        super().__init__(
            "FedDR",
            "FedDR",
            batch_size,
            num_epochs,
            lr,
            eta=eta,
            alpha=alpha,
        )


class FedDRServer(Server):
    """ """

    __name__ = "FedDRServer"

    def __init__(
        self,
        model: nn.Module,
        dataset: FedDataset,
        config: ServerConfig,
        client_config: ClientConfig,
    ) -> NoReturn:
        """ """
        super().__init__(model, dataset, config, client_config)
        self._regularizer = get_regularizer(
            self.config.reg_type,
            self.config.eta * self.config.num_clients / (self.config.num_clients + 1),
        )
        self._y_parameters = deepcopy(list(self.model.parameters()))  # y
        self._x_til_parameters = deepcopy(list(self.model.parameters()))  # x_tilde

    @property
    def client_cls(self) -> "Client":
        return FedDRClient

    @property
    def required_config_fields(self) -> List[str]:
        """ """
        return [
            "alpha",
            "eta",
            "reg_type",
        ]

    def communicate(self, target: "FedDRClient") -> NoReturn:
        """ """
        target._received_messages = {
            "parameters": deepcopy(list(self.model.parameters()))
        }

    def update(self) -> NoReturn:
        """ """
        # update y
        # FedDR paper Algorithm 1 line 7, first equation
        for yp, mp in zip(self._y_parameters, self.model.parameters()):
            yp.data.add_(mp.data - yp.data, alpha=self.config.alpha)

        # update x_tilde
        # FedDR paper Algorithm 1 line 7, second equation
        total_samples = sum([m["train_samples"] for m in self._received_messages])
        for m in self._received_messages:
            for i, xtp in enumerate(self._x_til_parameters):
                xtp.data.add_(
                    m["x_hat_delta"][i].data, alpha=m["train_samples"] / total_samples
                )

        # update server (global) model
        # FedDR paper Algorithm 1 line 8
        for mp, yp, xtp in zip(
            self.model.parameters(), self._y_parameters, self._x_til_parameters
        ):
            mp.data = (self._regularizer.coeff / self.config.eta) * xtp.data + (
                1 / (self.config.num_clients + 1)
            ) * yp.data
        for mp, p in zip(
            self.model.parameters(),
            self._regularizer.prox_eval(params=self.model.parameters()),
        ):
            mp.data = p.data


class FedDRClient(Client):
    """ """

    __name__ = "FedDRClient"

    def __init__(
        self,
        client_id: int,
        device: torch.device,
        model: nn.Module,
        dataset: FedDataset,
        config: ClientConfig,
    ) -> NoReturn:
        """ """
        super().__init__(client_id, device, model, dataset, config)
        self._y_parameters = None  # y
        self._x_hat_parameters = None  # x_hat
        self._x_hat_buffer = None  # x_hat_buffer

    @property
    def required_config_fields(self) -> List[str]:
        """ """
        return [
            "alpha",
            "eta",
        ]

    def communicate(self, target: "FedDRServer") -> NoReturn:
        """ """
        if self._x_hat_buffer is None:
            # outter iteration step -1, no need to communicate
            self._x_hat_buffer = deepcopy(self._x_hat_parameters)
            return
        x_hat_delta = [
            p.data - hp.data
            for p, hp in zip(self._x_hat_parameters, self._x_hat_buffer)
        ]
        self._x_hat_buffer = deepcopy(self._x_hat_parameters)
        target._received_messages.append(
            ClientMessage(
                **{
                    "client_id": self.client_id,
                    "x_hat_delta": x_hat_delta,
                    "train_samples": len(self.train_loader.dataset),
                    "metrics": self._metrics,
                }
            )
        )

    def update(self) -> NoReturn:
        """ """
        # copy the parameters from the server
        # x_bar
        try:
            self._cached_parameters = deepcopy(self._received_messages["parameters"])
        except KeyError:
            warnings.warn("No parameters received from server")
            warnings.warn("Using current model parameters as initial parameters")
            self._cached_parameters = deepcopy(list(self.model.parameters()))
        except Exception as err:
            raise err
        self._cached_parameters = [p.to(self.device) for p in self._cached_parameters]
        # update y
        if self._y_parameters is None:
            self._y_parameters = deepcopy(self._cached_parameters)
        else:
            for yp, cp, mp in zip(
                self._y_parameters, self._cached_parameters, self.model.parameters()
            ):
                yp.data.add_(cp.data - mp.data, alpha=self.config.alpha)
        # update x, via prox_sgd of y
        self.train()
        # update x_hat
        if self._x_hat_parameters is None:
            self._x_hat_parameters = deepcopy(self._cached_parameters)
        for hp, cp, mp in zip(
            self._x_hat_parameters, self._cached_parameters, self.model.parameters()
        ):
            hp.data = 2 * mp.data - yp.data

    def train(self) -> NoReturn:
        """ """
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
                    self.optimizer.step(self._y_parameters)
