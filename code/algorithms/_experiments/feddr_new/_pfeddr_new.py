"""
"""

from copy import deepcopy
import warnings
from typing import List, Dict

import torch  # noqa: F401

try:
    from tqdm.auto import tqdm  # noqa: F401
except ImportError:
    from tqdm import tqdm  # noqa: F401

from nodes import Server, Client, ServerConfig, ClientConfig  # noqa: F401
from optimizers import get_optimizer  # noqa: F401
from regularizers import get_regularizer  # noqa: F401
from data_processing.fed_dataset import FedDataset


class pFedDRServerConfig(ServerConfig):
    """ """

    __name__ = "pFedDRServerConfig"

    def __init__(
        self,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float,
        eta: float = 1.0,
        reg_type: str = "l1_norm",
    ) -> None:
        """ """
        super().__init__(
            "pFedDR",
            num_iters,
            num_clients,
            clients_sample_ratio,
            eta=eta,
            reg_type=reg_type,
        )


class pFedDRClientConfig(ClientConfig):
    """ """

    __name__ = "pFedDRClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-3,
        eta: float = 1.0,
        alpha: float = 1.9,
    ) -> None:
        """ """
        super().__init__(
            "pFedDR",
            "pFedDR",
            batch_size,
            num_epochs,
            lr,
            eta=eta,
            alpha=alpha,
        )


class pFedDRServer(Server):
    """ """

    __name__ = "pFedDRServer"

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: FedDataset,
        config: ServerConfig,
        client_config: ClientConfig,
    ) -> None:
        """ """
        super().__init__(model, dataset, config, client_config)
        raise NotImplementedError

    @property
    def client_cls(self) -> "Client":
        return pFedDRClient

    @property
    def required_config_fields(self) -> List[str]:
        """ """
        return [
            "alpha",
            "eta",
            "reg_type",
        ]

    def communicate(self, target: "pFedDRClient") -> None:
        """ """
        target._received_messages = {
            "parameters": deepcopy(list(self.model.parameters()))
        }
        self._num_communications += 1

    def update(self) -> None:
        """ """
        if len(self._received_messages) == 0:
            warnings.warn(
                "No message received from the clients, unable to update server model"
            )
            return

        raise NotImplementedError

        # clear received messages
        self._received_messages = []


class pFedDRClient(Client):
    """ """

    __name__ = "pFedDRClient"

    def __init__(
        self,
        client_id: int,
        device: torch.device,
        model: torch.nn.Module,
        dataset: FedDataset,
        config: ClientConfig,
    ) -> None:
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

    def communicate(self, target: "pFedDRServer") -> None:
        """ """
        raise NotImplementedError
        self._received_messages = {}

    def update(self) -> None:
        """ """
        raise NotImplementedError

    def train(self) -> None:
        """ """
        self.model.train()
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, part: str) -> Dict[str, float]:
        """ """
        assert part in [
            "train",
            "val",
        ]
        self.model.eval()
        _metrics = []
        data_loader = self.val_loader if part == "val" else self.train_loader
        for X, y in self.val_loader:
            logits = self.model(X)
            _metrics.append(self.dataset.evaluate(logits, y))
        metrics = {
            "num_examples": sum([m["num_examples"] for m in _metrics]),
        }
        for k in _metrics[0]:
            if k != "num_examples":  # average over all metrics
                metrics[k] = (
                    sum([m[k] * m["num_examples"] for m in _metrics])
                    / metrics["num_examples"]
                )
        return metrics
