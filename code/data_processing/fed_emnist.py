"""
federeated EMNIST
"""

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


__all__ = ["FedEMNIST",]


FED_EMNIST_DATA_DIR = CACHED_DATA_DIR / "fed_emnist"
FED_EMNIST_DATA_DIR.mkdir(exist_ok=True)


_label_mapping = {i: str(i) for i in range(10)}
_label_mapping.update({i+10:c for i,c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")})
_label_mapping.update({i+36:c for i,c in enumerate("abcdefghijklmnopqrstuvwxyz")})


class FedEMNIST(FedVisionDataset):
    """
    most methods in this class are modified from FedML
    """
    __name__ = "FedEMNIST"

    def _preload(self, datadir:Optional[Union[str,Path]]=None) -> NoReturn:
        """
        """
        self.datadir = Path(datadir or FED_EMNIST_DATA_DIR)

        self.DEFAULT_TRAIN_CLIENTS_NUM = 3400
        self.DEFAULT_TEST_CLIENTS_NUM = 3400
        self.DEFAULT_BATCH_SIZE = 20
        self.DEFAULT_TRAIN_FILE = "fed_emnist_train.h5"
        self.DEFAULT_TEST_FILE = "fed_emnist_test.h5"
        self._IMGAE = "pixels"

        self.criterion = torch.nn.CrossEntropyLoss()

        self.download_if_needed()

        #client id list
        train_file_path = self.datadir / self.DEFAULT_TRAIN_FILE
        test_file_path = self.datadir / self.DEFAULT_TEST_FILE
        with h5py.File(str(train_file_path), "r") as train_h5, h5py.File(str(test_file_path), "r") as test_h5:
            self._client_ids_train = list(train_h5[self._EXAMPLE].keys())
            self._client_ids_test = list(test_h5[self._EXAMPLE].keys())
            self._n_class = len(np.unique([
                train_h5[self._EXAMPLE][self._client_ids_train[idx]][self._LABEL][0] \
                    for idx in range(self.DEFAULT_TRAIN_CLIENTS_NUM)
            ]))

    def get_dataloader(self,
                       train_bs:int,
                       test_bs:int,
                       client_idx:Optional[int]=None,) -> Tuple[data.DataLoader, data.DataLoader]:
        """
        """
        train_h5 = h5py.File(str(self.datadir / self.DEFAULT_TRAIN_FILE), "r")
        test_h5 = h5py.File(str(self.datadir / self.DEFAULT_TEST_FILE), "r")
        train_x, train_y, test_x, test_y = [], [], [], []
        
        # load data
        if client_idx is None:
            # get ids of all clients
            train_ids = self._client_ids_train
            test_ids = self._client_ids_test
        else:
            # get ids of single client
            train_ids = [self._client_ids_train[client_idx]]
            test_ids = [self._client_ids_test[client_idx]]

        # load data in numpy format from h5 file
        train_x = np.vstack([train_h5[self._EXAMPLE][client_id][self._IMGAE][()] for client_id in train_ids])
        train_y = np.concatenate([train_h5[self._EXAMPLE][client_id][self._LABEL][()] for client_id in train_ids])
        test_x = np.vstack([test_h5[self._EXAMPLE][client_id][self._IMGAE][()] for client_id in test_ids])
        test_y = np.concatenate([test_h5[self._EXAMPLE][client_id][self._LABEL][()] for client_id in test_ids])

        # dataloader
        train_ds = data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.long)))
        train_dl = data.DataLoader(dataset=train_ds,
                                   batch_size=train_bs,
                                   shuffle=True,
                                   drop_last=False)

        test_ds = data.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y.astype(np.long)))
        test_dl = data.DataLoader(dataset=test_ds,
                                  batch_size=test_bs,
                                  shuffle=True,
                                  drop_last=False)

        train_h5.close()
        test_h5.close()
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
        return "https://fedml.s3-us-west-1.amazonaws.com/fed_emnist.tar.bz2"
