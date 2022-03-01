"""
"""

import random
from abc import ABC, abstractmethod
from itertools import repeat
from copy import deepcopy
from typing import Any, Optional, NoReturn, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

from data_processing.fed_dataset import FedDataset
from .optimizer import get_optimizer


__all__ = [
    "Server", "Client",
]


class SeverConfig:
    """
    """
    def __init__(self,
                 algorithm:str,
                 num_iters:int,
                 num_clients:int,
                 clients_sample_ratio:float) -> NoReturn:
        """
        """
        self.algorithm = algorithm
        self.num_iters = num_iters
        self.num_clients = num_clients
        self.clients_sample_ratio = clients_sample_ratio


class ClientConfig:
    """
    """
    def __init__(self,
                 client_cls:str,
                 algorithm:str,
                 optimizer:str,
                 batch_size:int,
                 lr:float,
                 num_epochs:int,) -> NoReturn:
        """
        """
        self.client_cls = client_cls
        self.algorithm = algorithm
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs


class Node(ABC):
    """
    """

    @abstractmethod
    def communicate(self, target:"Node") -> NoReturn:
        """
        communicate model parameters, gradients, etc. to `target` node
        """
        raise NotImplementedError

    @abstractmethod
    def update(self) -> NoReturn:
        """
        update model parameters, gradients, etc.
        according to `self._reveived_messages`
        """
        raise NotImplementedError


class Server(Node):
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

        self._clients = self._setup_clients(dataset, client_config)

        self._received_messages = []

    def _setup_clients(self, dataset:FedDataset, client_config:ClientConfig) -> List[Node]:
        """
        setup clients
        """
        client_cls = eval(client_config.client_cls)
        return [
            client_cls(client_id, device, deepcopy(self.model), deepcopy(self.criterion), dataset, client_config) \
                for client_id, device in zip(range(self.config.num_clients), self._allocate_devices())
        ]

    def _allocate_devices(self) -> List[torch.device]:
        """
        allocate devices for clients, can be used in `_setup_clients`
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            return list(repeat(torch.device("cpu"), self.config.num_clients))
        return [torch.device(f"cuda:{i%num_gpus}") for i in range(self.config.num_clients)]

    def _sample_clients(self) -> List[int]:
        """
        sample clients for each iteration
        """
        k = int(self.config.num_clients * self.config.clients_sample_ratio)
        return random.sample(range(self.config.num_clients), k)

    def train(self) -> NoReturn:
        """
        """
        with tqdm(range(self.config.num_iters), total=self.config.num_iters) as pbar:
            for i in pbar:
                for client_id in self._sample_clients():
                    client = self._clients[client_id]
                    self.communicate(client)
                    client.update()
                    client.communicate(self)
                self.update()


class Client(Node):
    """
    """
    def __init__(self,
                 client_id:int,
                 device:torch.device,
                 model:nn.Module,
                 criterion:nn.Module,
                 dataset:FedDataset,
                 config:ClientConfig,) -> NoReturn:
        """
        """
        self.client_id = client_id
        self.device = device
        self.model = model
        self.criterion = criterion
        self.dataset = dataset
        self.config = config

        self._optimizer = get_optimizer(
            optimizer_name=config.optimizer, params=self.model.parameters(), config=config
        )
        self.train_loader, self.val_loader = \
            self.dataset.get_data_loaders(self.config.batch_size, self.config.batch_size, self.client_id)

        self._global_model = None
        self._received_messages = {}

    def train(self) -> NoReturn:
        """
        """
        self.model.train()
        epoch_losses = []
        for epoch in range(self.config.num_epochs):
            batch_losses = []
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self._optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self._optimizer.step()
                batch_losses.append(loss.item())
            epoch_losses.append(sum(batch_losses) / len(batch_losses))

    @abstractmethod
    def evaluate(self) -> NoReturn:
        """
        """
        raise NotImplementedError

    def set_parameters(self, model:nn.Module) -> NoReturn:
        """
        """
        self.update_parameters(new_params=model.parameters())

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
