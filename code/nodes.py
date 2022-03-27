"""
"""

import random, warnings
from abc import ABC, abstractmethod
from itertools import repeat
from copy import deepcopy
from collections import defaultdict
from typing import Any, Optional, NoReturn, Iterable, List, Sequence, Tuple, Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from torch.optim import Optimizer, SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR, OneCycleLR

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm
from easydict import EasyDict as ED

from misc import ReprMixin
from loggers import LoggerManager
from data_processing.fed_dataset import FedDataset
from algorithms.optimizers import get_optimizer


__all__ = [
    "Server",
    "Client",
    "ServerConfig",
    "ClientConfig",
    "ClientMessage",
]


class ServerConfig(ReprMixin):
    """ """

    __name__ = "ServerConfig"

    def __init__(
        self,
        algorithm: str,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        txt_logger: bool = True,
        csv_logger: bool = True,
        eval_every: int = 1,
        **kwargs: Any,
    ) -> NoReturn:
        """ """
        self.algorithm = algorithm
        self.num_iters = num_iters
        self.num_clients = num_clients
        self.clients_sample_ratio = clients_sample_ratio
        self.txt_logger = txt_logger
        self.csv_logger = csv_logger
        self.eval_every = eval_every
        for k, v in kwargs.items():
            setattr(self, k, v)

    def extra_repr_keys(self) -> List[str]:
        """ """
        return super().extra_repr_keys() + list(self.__dict__)


class ClientConfig(ReprMixin):
    """ """

    __name__ = "ClientConfig"

    def __init__(
        self,
        algorithm: str,
        optimizer: str,
        batch_size: int,
        num_epochs: int,
        lr: float,
        **kwargs: Any,
    ) -> NoReturn:
        """ """
        self.algorithm = algorithm
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        for k, v in kwargs.items():
            setattr(self, k, v)

    def extra_repr_keys(self) -> List[str]:
        """ """
        return super().extra_repr_keys() + list(self.__dict__)


class Node(ReprMixin, ABC):
    """ """

    __name__ = "Node"

    @abstractmethod
    def communicate(self, target: "Node") -> NoReturn:
        """
        communicate model parameters, gradients, etc. to `target` node
        for example, for a client node, communicate model parameters to server node via
        ```python
        target._received_messages.append(
            {
                "parameters": deepcopy(list(self.model.parameters())),
                "train_samples": self.config.num_epochs * self.config.num_steps * self.config.batch_size,
            }
        )
        ```
        for a server node, communicate model parameters to clients via
        ```python
        target._received_messages = {"parameters": deepcopy(list(self.model.parameters()))}
        ```python
        """
        raise NotImplementedError

    @abstractmethod
    def update(self) -> NoReturn:
        """
        update model parameters, gradients, etc.
        according to `self._reveived_messages`
        """
        raise NotImplementedError

    def _post_init(self) -> NoReturn:
        """
        check if all required field in the config are set
        """
        return all([hasattr(self.config, k) for k in self.required_config_fields])

    @property
    @abstractmethod
    def required_config_fields(self) -> List[str]:
        """ """
        raise NotImplementedError


class Server(Node):
    """ """

    __name__ = "Server"

    def __init__(
        self,
        model: nn.Module,
        dataset: FedDataset,
        config: ServerConfig,
        client_config: ClientConfig,
    ) -> NoReturn:
        """ """
        self.model = model
        self.dataset = dataset
        self.criterion = deepcopy(dataset.criterion)
        self.config = config
        self.device = torch.device("cpu")

        self._clients = self._setup_clients(dataset, client_config)
        logger_config = dict(
            txt_logger=self.config.txt_logger,
            csv_logger=self.config.csv_logger,
            algorithm=self.config.algorithm,
            model=self.model.__class__.__name__,
            dataset=dataset.__class__.__name__,
        )
        self._logger_manager = LoggerManager.from_config(logger_config)

        # set batch_size, in case of centralized training
        setattr(self.config, "batch_size", client_config.batch_size)

        self._received_messages = []
        self._num_communications = 0

        self.n_iter = 0

        self._post_init()

    def _setup_clients(
        self, dataset: FedDataset, client_config: ClientConfig
    ) -> List[Node]:
        """
        setup clients
        """
        print(f"setup clients...")
        return [
            self.client_cls(
                client_id, device, deepcopy(self.model), dataset, client_config
            )
            for client_id, device in zip(
                range(self.config.num_clients), self._allocate_devices()
            )
        ]

    def _allocate_devices(self) -> List[torch.device]:
        """
        allocate devices for clients, can be used in `_setup_clients`
        """
        print(f"allocate devices...")
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            return list(repeat(torch.device("cpu"), self.config.num_clients))
        return [
            torch.device(f"cuda:{i%num_gpus}") for i in range(self.config.num_clients)
        ]

    def _sample_clients(self) -> List[int]:
        """
        sample clients for each iteration
        """
        k = int(self.config.num_clients * self.config.clients_sample_ratio)
        return random.sample(range(self.config.num_clients), k)

    def _communicate(self, target: "Client") -> NoReturn:
        """broadcast to target client, and maintain state variables"""
        self.communicate(target)
        self._num_communications += 1

    def _update(self) -> NoReturn:
        """server update, and clear cached messages from clients of the previous iteration"""
        self._logger_manager.log_message("Server update...")
        if len(self._received_messages) == 0:
            warnings.warn(
                "No message received from the clients, unable to update server model"
            )
            return
        assert all(
            [isinstance(m, ClientMessage) for m in self._received_messages]
        ), "received messages must be of type `ClientMessage`"
        self.update()
        self._received_messages = []
        self._logger_manager.log_message("Server update finished...")

    def train(
        self, mode: str = "federated", extra_configs: Optional[dict] = None
    ) -> NoReturn:
        """ """
        if mode == "federated":
            self.train_federated(extra_configs)
        elif mode == "centralized":
            self.train_centralized(extra_configs)
        else:
            raise ValueError(f"mode {mode} is not supported")

    def train_centralized(self, extra_configs: Optional[dict] = None) -> NoReturn:
        """ """
        self._logger_manager.log_message("Training centralized...")
        extra_configs = ED(extra_configs or {})

        batch_size = extra_configs.get("batch_size", self.config.batch_size)
        train_loader, val_loader = self.dataset.get_dataloader(
            batch_size, batch_size, None
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.train()
        self.model.to(device)

        criterion = deepcopy(self.dataset.criterion)
        lr = extra_configs.get("lr", 1e-2)
        optimizer = extra_configs.get("optimizer", SGD(self.model.parameters(), lr))
        scheduler = extra_configs.get(
            "scheduler", LambdaLR(optimizer, lambda epoch: 1 / (epoch + 1))
        )

        epoch_losses = []
        self.n_iter, global_step = 0, 0
        for self.n_iter in range(self.config.num_iters):
            with tqdm(
                total=len(train_loader.dataset),
                desc=f"Epoch {self.n_iter+1}/{self.config.num_iters}",
                unit="sample",
            ) as pbar:
                epoch_loss = []
                batch_losses = []
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    output = self.model(data)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    batch_losses.append(loss.item())
                    optimizer.step()
                    global_step += 1
                    pbar.set_postfix(
                        **{
                            "loss (batch)": loss.item(),
                            "lr": scheduler.get_last_lr()[0],
                        }
                    )
                    pbar.update(data.shape[0])
                epoch_loss.append(sum(batch_losses) / len(batch_losses))
                if (self.n_iter + 1) % self.config.eval_every == 0:
                    print("evaluating...")
                    metrics = self.evaluate_centralized(val_loader)
                    self._logger_manager.log_metrics(
                        None,
                        metrics,
                        step=global_step,
                        epoch=self.n_iter + 1,
                        part="val",
                    )
                    metrics = self.evaluate_centralized(train_loader)
                    self._logger_manager.log_metrics(
                        None,
                        metrics,
                        step=global_step,
                        epoch=self.n_iter + 1,
                        part="train",
                    )
                scheduler.step()

        self.model.to(self.device)  # move to the original device
        self._logger_manager.log_message("Centralized training finished...")

    def train_federated(self, extra_configs: Optional[dict] = None) -> NoReturn:
        """
        TODO: run clients in parallel
        """
        self._logger_manager.log_message("Training federated...")
        self.n_iter = 0
        for self.n_iter in range(self.config.num_iters):
            selected_clients = self._sample_clients()
            with tqdm(
                total=len(selected_clients),
                desc=f"Iter {self.n_iter+1}/{self.config.num_iters}",
                unit="client",
            ) as pbar:
                for client_id in selected_clients:
                    client = self._clients[client_id]
                    self._communicate(client)
                    client._update()
                    # if client_id in chosen_clients:
                    #     client._communicate(self)
                    if (self.n_iter + 1) % self.config.eval_every == 0:
                        for part in self.dataset.data_parts:
                            metrics = client.evaluate(part)
                            # print(f"metrics: {metrics}")
                            self._logger_manager.log_metrics(
                                client_id,
                                metrics,
                                step=self.n_iter + 1,
                                epoch=self.n_iter + 1,
                                part=part,
                            )
                    client._communicate(self)
                    pbar.update(1)
                if (self.n_iter + 1) % self.config.eval_every == 0:
                    self.aggregate_client_metrics()
                self._update()
        self._logger_manager.log_message("Federated training finished...")

    def evaluate_centralized(self, dataloader: DataLoader) -> Dict[str, float]:
        """ """
        metrics = []
        for (X, y) in dataloader:
            X, y = X.to(self.model.device), y.to(self.model.device)
            probs = self.model(X)
            metrics.append(self.dataset.evaluate(probs, y))
        num_samples = sum([m["num_samples"] for m in metrics])
        metrics_names = [k for k in metrics[0] if k != "num_samples"]
        metrics = {
            k: sum([m[k] * m["num_samples"] for m in metrics]) / num_samples
            for k in metrics_names
        }
        metrics["num_samples"] = num_samples
        return metrics

    def aggregate_client_metrics(self) -> NoReturn:
        """ """
        if not any(["metrics" in m for m in self._received_messages]):
            raise ValueError("no metrics received from clients")
        for part in self.dataset.data_parts:
            metrics = defaultdict(float)
            for m in self._received_messages:
                if "metrics" not in m:
                    continue
                for k, v in m["metrics"][part].items():
                    if k != "num_samples":
                        metrics[k] += (
                            m["metrics"][part][k] * m["metrics"][part]["num_samples"]
                        )
                    else:
                        metrics[k] += m["metrics"][part][k]
            for k in metrics:
                if k != "num_samples":
                    metrics[k] /= metrics["num_samples"]
            self._logger_manager.log_metrics(
                None,
                dict(metrics),
                step=self.n_iter + 1,
                epoch=self.n_iter + 1,
                part=part,
            )

    def add_parameters(self, params: Iterable[Parameter], ratio: float) -> NoReturn:
        """ """
        for server_param, param in zip(self.model.parameters(), params):
            server_param.data.add_(
                param.data.detach().clone().to(self.device), alpha=ratio
            )

    def extra_repr_keys(self) -> List[str]:
        """ """
        return super().extra_repr_keys() + [
            "config",
            "client_config",
        ]

    @property
    @abstractmethod
    def client_cls(self) -> "Client":
        """ """
        raise NotImplementedError


class Client(Node):
    """ """

    __name__ = "Client"

    def __init__(
        self,
        client_id: int,
        device: torch.device,
        model: nn.Module,
        dataset: FedDataset,
        config: ClientConfig,
    ) -> NoReturn:
        """ """
        self.client_id = client_id
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.criterion = deepcopy(dataset.criterion)
        self.dataset = dataset
        self.config = config

        self.optimizer = get_optimizer(
            optimizer_name=config.optimizer,
            params=self.model.parameters(),
            config=config,
        )
        self.train_loader, self.val_loader = self.dataset.get_dataloader(
            self.config.batch_size, self.config.batch_size, self.client_id
        )

        self._cached_parameters = None
        self._received_messages = {}
        self._metrics = {}

        self._post_init()

    def _communicate(self, target: "Server") -> NoReturn:
        """send messages to the server, and maintain state variables"""
        self.communicate(target)
        target._num_communications += 1
        self._metrics = {}

    def _update(self) -> NoReturn:
        """client update, and clear cached messages from the server of the previous iteration"""
        self.update()
        self._received_messages = {}

    @abstractmethod
    def train(self) -> NoReturn:
        """
        main part of inner loop solver, using the data from dataloaders

        basic example:
        ```python
        self.model.train()
        epoch_losses = []
        for epoch in range(self.config.num_epochs):
            batch_losses = []
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())
            epoch_losses.append(sum(batch_losses) / len(batch_losses))
        ```
        """
        raise NotImplementedError

    def solve_inner(self) -> NoReturn:
        """
        alias of `train`
        """
        self.train()

    def sample_data(self) -> Tuple[Tensor, Tensor]:
        """
        sample data for training
        """
        return next(iter(self.train_loader))

    @torch.no_grad()
    def evaluate(self, part: str) -> Dict[str, float]:
        """ """
        assert part in self.dataset.data_parts, "Invalid part name"
        self.model.eval()
        _metrics = []
        data_loader = self.val_loader if part == "val" else self.train_loader
        for X, y in data_loader:
            X, y = X.to(self.device), y.to(self.device)
            logits = self.model(X)
            _metrics.append(self.dataset.evaluate(logits, y))
        self._metrics[part] = {
            "num_samples": sum([m["num_samples"] for m in _metrics]),
        }
        for k in _metrics[0]:
            if k != "num_samples":  # average over all metrics
                self._metrics[part][k] = (
                    sum([m[k] * m["num_samples"] for m in _metrics])
                    / self._metrics[part]["num_samples"]
                )
        return self._metrics[part]

    def get_parameters(self) -> Iterable[Parameter]:
        """ """
        return self.model.parameters()

    def set_parameters(self, params: Iterable[Parameter]) -> NoReturn:
        """ """
        for client_param, param in zip(self.model.parameters(), params):
            client_param.data = param.data.detach().clone().to(self.device)

    def get_gradients(self) -> List[Tensor]:
        """ """
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def extra_repr_keys(self) -> List[str]:
        """ """
        return super().extra_repr_keys() + [
            "client_id",
            "config",
        ]


class ClientMessage(dict):
    """
    a class used to specify required fields for a message from client to server
    """

    __name__ = "ClientMessage"

    def __init__(
        self, client_id: int, train_samples: int, metrics: dict, **kwargs
    ) -> NoReturn:
        """ """
        super().__init__(
            client_id=client_id, train_samples=train_samples, metrics=metrics, **kwargs
        )
