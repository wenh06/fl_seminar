"""
(part of) federeated EMNIST used in the FedDR paper
"""

import json
from pathlib import Path
from typing import NoReturn, Optional, Union, List, Callable, Tuple, Dict, Sequence

import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from misc import CACHED_DATA_DIR
from models.utils import top_n_accuracy
from .fed_dataset import FedVisionDataset


__all__ = ["FedDR_FEMNIST",]


FEDDR_FEMNIST_DATA_DIR = CACHED_DATA_DIR / "feddr_femnist"
FEDDR_FEMNIST_DATA_DIR.mkdir(exist_ok=True)
_label_mapping = {i:c for i,c in enumerate("abcdefghijklmnopqrstuvwxyz"[:10])}


class FedDR_FEMNIST(FedVisionDataset):
    """
    """
    __name__ = "FedDR_FEMNIST"

    def _preload(self, datadir:Optional[Union[str,Path]]=None) -> NoReturn:
        """
        """
        self.datadir = Path(datadir or FEDDR_FEMNIST_DATA_DIR)

        self.DEFAULT_TRAIN_CLIENTS_NUM = 200
        self.DEFAULT_TEST_CLIENTS_NUM = 200
        self.DEFAULT_BATCH_SIZE = 20
        self.DEFAULT_TRAIN_FILE = Path("train") / Path("mytrain.json")
        self.DEFAULT_TEST_FILE = Path("test") / Path("mytest.json")

        self._EXAMPLE = "user_data"
        self._IMGAE = "x"
        self._LABEL = "y"

        self.criterion = torch.nn.CrossEntropyLoss()

        self.download_if_needed()

        #client id list
        train_file_path = self.datadir / self.DEFAULT_TRAIN_FILE
        test_file_path = self.datadir / self.DEFAULT_TEST_FILE
        train_data_dict = json.loads(train_file_path.read_text())
        test_data_dict = json.loads(test_file_path.read_text())
        self._client_ids_train = train_data_dict["users"]
        self._client_ids_test = test_data_dict["users"]
            
        self._n_class = len(np.unique(np.concatenate([
            train_data_dict[self._EXAMPLE][self._client_ids_train[idx]][self._LABEL] \
                for idx in range(self.DEFAULT_TRAIN_CLIENTS_NUM)
        ])))
        del train_data_dict
        del test_data_dict

    def get_dataloader(self,
                       train_bs:int,
                       test_bs:int,
                       client_idx:Optional[int]=None,) -> Tuple[data.DataLoader, data.DataLoader]:
        """
        """
        train_file_path = self.datadir / self.DEFAULT_TRAIN_FILE
        test_file_path = self.datadir / self.DEFAULT_TEST_FILE
        train_data_dict = json.loads(train_file_path.read_text())
        test_data_dict = json.loads(test_file_path.read_text())

        # load data
        if client_idx is None:
            # get ids of all clients
            train_ids = self._client_ids_train
            test_ids = self._client_ids_test
        else:
            # get ids of single client
            train_ids = [self._client_ids_train[client_idx]]
            test_ids = [self._client_ids_test[client_idx]]

        # load data
        train_x = np.vstack([train_data_dict[self._EXAMPLE][client_id][self._IMGAE] for client_id in train_ids])
        train_y = np.concatenate([train_data_dict[self._EXAMPLE][client_id][self._LABEL] for client_id in train_ids])
        test_x = np.vstack([test_data_dict[self._EXAMPLE][client_id][self._IMGAE] for client_id in test_ids])
        test_y = np.concatenate([test_data_dict[self._EXAMPLE][client_id][self._LABEL] for client_id in test_ids])

        # dataloader
        train_ds = data.TensorDataset(
            torch.from_numpy(train_x.reshape((-1,28,28))).unsqueeze(1),
            torch.from_numpy(train_y.astype(np.long))
        )
        train_dl = data.DataLoader(dataset=train_ds,
                                   batch_size=train_bs,
                                   shuffle=True,
                                   drop_last=False)

        test_ds = data.TensorDataset(
            torch.from_numpy(test_x.reshape((-1,28,28))).unsqueeze(1),
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
        return "http://218.245.5.12/NLP/federated/feddr-femnist.zip"
