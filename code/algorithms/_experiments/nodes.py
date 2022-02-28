"""
"""

from dataclasses import dataclass
from copy import deepcopy
from typing import Optional, NoReturn, Iterable, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter

from ...data_processing.fed_dataset import FedDataset


__all__ = [
    "Server", "Client",
]


@dataclass
class SeverConfig:
    """
    """
    num_iters: int
    num_clients: int
    num_selected_clients: Optional[int] = None


@dataclass
class ClientConfig:
    """
    """
    batch_size: int
    learning_rate: float
    num_epochs: int


class Server:
    """
    """
    def __init__(self,
                 model:nn.Module,
                 criterion:nn.Module,
                 dataset:FedDataset,
                 config:SeverConfig,
                 client_config:ClientConfig,) -> NoReturn:
        """
        """
        self.model = model
        self.criterion = criterion
        self.config = config

        self._clients = [
            Client(client_id, deepcopy(model), deepcopy(criterion), dataset, client_config) \
                for client_id in range(config.num_clients)
        ]

    def train(self) -> NoReturn:
        """
        """
        raise NotImplementedError

    def add_grad(self, client:Client, ratio:float) -> NoReturn:
        """
        """
        for param, client_grad in zip(self.model.parameters(), client.get_grads()):
            if param.grad is not None and client_grad is not None:
                param.grad = param.grad + client_grad.clone() * ratio

    def aggregate_grads(self) -> NoReturn:
        """
        """
        raise NotImplementedError

    def send_parameters(self):
        """
        """
        for c in self.clients:
            c.set_parameters(self.model)

    def add_parameters(self, user, ratio):
        """
        """
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        raise NotImplementedError


class Client:
    """
    """
    def __init__(self,
                 client_id:int,
                 model:nn.Module,
                 criterion:nn.Module,
                 dataset:FedDataset,
                 config:ClientConfig,) -> NoReturn:
        """
        """
        self.client_id = client_id
        self.model = model
        self.criterion = criterion
        self.dataset = dataset
        self.config = config

        self.train_loader, self.val_loader = \
            self.dataset.get_data_loaders(self.config.batch_size, self.config.batch_size, self.client_id)

    def train(self) -> NoReturn:
        """
        """
        raise NotImplementedError


    def evaluate(self) -> NoReturn:
        """
        """
        raise NotImplementedError

    def set_parameters(self, model:nn.Module) -> NoReturn:
        """
        """
        raise NotImplementedError

    def update_parameters(self, new_params:Iterable[Parameter]) -> NoReturn:
        """
        """
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self) -> List[Tensor]:
        """
        """
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(None)
            else:
                grads.append(param.grad.data)
        return grads
