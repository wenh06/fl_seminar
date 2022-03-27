"""
"""

from copy import deepcopy
import warnings
from typing import List, NoReturn, Dict

import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

from nodes import Server, Client, ServerConfig, ClientConfig
from ..optimizers import get_optimizer
from ..regularizers import get_regularizer


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
    ) -> NoReturn:
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
    ) -> NoReturn:
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
        model: nn.Module,
        dataset: FedDataset,
        config: ServerConfig,
        client_config: ClientConfig,
    ) -> NoReturn:
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

    def communicate(self, target: "pFedDRClient") -> NoReturn:
        """ """
        target._received_messages = {
            "parameters": deepcopy(list(self.model.parameters()))
        }
        self._num_communications += 1

    def update(self) -> NoReturn:
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
        model: nn.Module,
        dataset: FedDataset,
        config: ClientConfig,
    ) -> NoReturn:
        """ """
        super().__init__(client_id, device, model, dataset, config)
        raise NotImplementedError

    @property
    def required_config_fields(self) -> List[str]:
        """ """
        return [
            "alpha",
            "eta",
        ]

    def communicate(self, target: "pFedDRServer") -> NoReturn:
        """ """
        raise NotImplementedError
        self._received_messages = {}

    def update(self) -> NoReturn:
        """ """
        raise NotImplementedError

    def train(self) -> NoReturn:
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
