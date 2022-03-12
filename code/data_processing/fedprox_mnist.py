"""
(part of) federeated EMNIST used in the FedProx paper

References
----------
1. https://github.com/litian96/FedProx/tree/master/data/mnist
2. https://github.com/litian96/FedProx/blob/master/data/mnist/generate_niid.py
"""

from pathlib import Path
from typing import NoReturn, Optional, Union, List, Callable, Tuple, Dict, Sequence

import numpy as np
from scipy.io import loadmat
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from misc import CACHED_DATA_DIR
from models.utils import top_n_accuracy
from .fed_dataset import FedVisionDataset


__all__ = ["FedProxMNIST",]


FEDPROX_FEMNIST_DATA_DIR = CACHED_DATA_DIR / "fedprox_mnist"
FEDPROX_FEMNIST_DATA_DIR.mkdir(exist_ok=True)
# _label_mapping = {i:c for i,c in enumerate("abcdefghijklmnopqrstuvwxyz"[:10])}


class FedProxMNIST(FedVisionDataset):
    """
    """
    __name__ = "FedProxMNIST"

    def _preload(self, datadir:Optional[Union[str,Path]]=None) -> NoReturn:
        """
        """
        self.datadir = Path(datadir or FEDPROX_FEMNIST_DATA_DIR)

        self.DEFAULT_TRAIN_CLIENTS_NUM = 1000
        self.DEFAULT_TEST_CLIENTS_NUM = 1000
        self.DEFAULT_BATCH_SIZE = 20

        self.DEFAULT_TRAIN_FILE = "fedprox-mnist.mat"
        self.DEFAULT_TEST_FILE = "fedprox-mnist.mat"

        self._EXAMPLE = ""
        self._IMGAE = "data"
        self._LABEL = "label"

        self.criterion = torch.nn.CrossEntropyLoss()

        self.download_if_needed()
        self._client_data = generate_niid(loadmat(self.datadir / self.DEFAULT_TRAIN_FILE))

        self._client_ids_train = list(range(self.DEFAULT_TRAIN_CLIENTS_NUM))
        self._client_ids_test = list(range(self.DEFAULT_TEST_CLIENTS_NUM))

    def get_dataloader(self,
                       train_bs:int,
                       test_bs:int,
                       client_idx:Optional[int]=None,) -> Tuple[data.DataLoader, data.DataLoader]:
        """
        """
        if client_idx is None:
            # get ids of all clients
            train_ids = self._client_ids_train
            test_ids = self._client_ids_test
        else:
            # get ids of single client
            train_ids = [self._client_ids_train[client_idx]]
            test_ids = [self._client_ids_test[client_idx]]

        # load data
        train_x = np.vstack([self._client_data[client_id]["train_x"] for client_id in train_ids])
        train_y = np.concatenate([self._client_data[client_id]["train_y"] for client_id in train_ids])
        test_x = np.vstack([self._client_data[client_id]["test_x"] for client_id in test_ids])
        test_y = np.concatenate([self._client_data[client_id]["test_y"] for client_id in test_ids])

        # dataloader
        train_ds = data.TensorDataset(
            torch.from_numpy(train_x).unsqueeze(1),
            torch.from_numpy(train_y.astype(np.long))
        )
        train_dl = data.DataLoader(dataset=train_ds,
                                   batch_size=train_bs,
                                   shuffle=True,
                                   drop_last=False)

        test_ds = data.TensorDataset(
            torch.from_numpy(test_x).unsqueeze(1),
            torch.from_numpy(test_y.astype(np.long))
        )
        test_dl = data.DataLoader(dataset=test_ds,
                                  batch_size=test_bs,
                                  shuffle=True,
                                  drop_last=False)

        return train_dl, test_dl

    def extra_repr_keys(self) -> List[str]:
        """
        """
        return ["n_class",] + super().extra_repr_keys()

    def get_class(self, label:torch.Tensor) -> str:
        """
        """
        return _label_mapping[label.item()]

    def get_classes(self, labels:torch.Tensor) -> List[str]:
        return [_label_mapping[l] for l in labels.cpu().numpy()]

    def evaluate(self, probs:torch.Tensor, truths:torch.Tensor) -> Dict[str, float]:
        """
        """
        return {
            "acc": top_n_accuracy(probs, truths, 1),
            "top3_acc": top_n_accuracy(probs, truths, 3),
            "top5_acc": top_n_accuracy(probs, truths, 5),
            "loss": self.criterion(probs, truths).item(),
            "num_samples": probs.shape[0],
        }

    @property
    def url(self) -> str:
        # https://drive.google.com/file/d/1tCEcJgRJ8NdRo11UJZR6WSKMNdmox4GC/view?usp=sharing
        return "http://218.245.5.12/NLP/federated/fedprox-mnist.zip"


def generate_niid(mnist_data:Dict[str, np.ndarray],
                  num_clients:int=1000,
                  lower_bound:int=10,
                  class_per_client:int=2,
                  seed:int=42,
                  train_ratio:float=0.9,) -> Dict[str, np.ndarray]:
    """
    modified from
    https://github.com/litian96/FedProx/blob/master/data/mnist/generate_niid.py
    """
    NUM_CLASSES = 10
    IMG_SHAPE = (28, 28)
    mnist_data["data"] = (mnist_data["data"] / 255.0).astype(np.float32)
    eps = 1e-5
    options = dict(axis=0, keepdims=True)
    mnist_data["data"] = \
        (mnist_data["data"] - mnist_data["data"].mean(**options)) / (mnist_data["data"].std(**options) + eps)
    mnist_data["data"] = mnist_data["data"].T.reshape((-1, *IMG_SHAPE))
    mnist_data["label"] = mnist_data["label"].flatten()

    class_inds = {
        i: np.where(mnist_data["label"] == i)[0] for i in range(NUM_CLASSES)
    }
    class_nums = [lower_bound//class_per_client for _ in range(class_per_client-1)]
    class_nums.append(lower_bound - sum(class_nums))

    clients_data = [
        {
            k: np.empty((0, *IMG_SHAPE), dtype=np.float32) \
                if k.startswith("train") else np.array([], dtype=np.int64) \
                    for k in ["train_x", "train_y", "test_x", "test_y",]
        } for _ in range(num_clients)
    ]
    # idx = np.zeros(NUM_CLASSES, dtype=np.int64)
    idx = {i: 0 for i in range(NUM_CLASSES)}
    for c in range(num_clients):
        for j, n in enumerate(class_nums):
            label = (c + j) % NUM_CLASSES
            inds = class_inds[label][idx[label]: idx[label] + n]
            clients_data[c]["train_x"] = \
                np.append(clients_data[c]["train_x"], mnist_data["data"][inds, ...], axis=0)
            clients_data[c]["train_y"] = \
                np.append(clients_data[c]["train_y"], np.full_like(inds, label, dtype=np.int64))
            idx[label] += n
    print(f"idx = {idx}")
    print(f"class_inds = {[(l, len(class_inds[l])) for l in range(NUM_CLASSES)]}")

    rng = np.random.default_rng(seed)
    probs = rng.lognormal(0, 2.0, (NUM_CLASSES, num_clients//NUM_CLASSES, class_per_client))
    probs = np.array([[[len(class_inds[i])-idx[i]]] for i in range(NUM_CLASSES)]) \
        * probs / probs.sum(axis=(1,2), keepdims=True)
    for c in range(num_clients):
        for j, n in enumerate(class_nums):
            label = (c + j) % NUM_CLASSES
            num_samples = round(probs[label, c//NUM_CLASSES, j])
            if idx[label] + num_samples < len(class_inds[label]):
                inds = class_inds[label][idx[label]: idx[label] + num_samples]
                clients_data[c]["train_x"] = \
                    np.append(clients_data[c]["train_x"], mnist_data["data"][inds, ...], axis=0)
                clients_data[c]["train_y"] = \
                    np.append(clients_data[c]["train_y"], np.full_like(inds, label, dtype=np.int64))
                idx[label] += num_samples
        num_samples = clients_data[c]["train_x"].shape[0]
        inds = rng.choice(num_samples, num_samples, replace=False)
        train_len = int(train_ratio * num_samples)
        clients_data[c]["test_x"] = clients_data[c]["train_x"][inds[train_len:], ...]
        clients_data[c]["test_y"] = clients_data[c]["train_y"][inds[train_len:]]
        clients_data[c]["train_x"] = clients_data[c]["train_x"][inds[:train_len], ...]
        clients_data[c]["train_y"] = clients_data[c]["train_y"][inds[:train_len]]
    print(f"idx = {idx}")

    return clients_data
