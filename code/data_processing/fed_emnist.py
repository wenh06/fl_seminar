"""
federeated EMNIST
"""

import os
from typing import NoReturn, Optional, Union, List, Callable, Tuple, Dict, Sequence

import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from misc import CACHED_DATA_DIR, default_class_repr
from .fed_dataset import FedVisionDataset


__all__ = ["FedEMNIST",]


FED_EMNIST_DATA_DIR = os.path.join(CACHED_DATA_DIR, "fed_emnist")


class FedEMNIST(FedVisionDataset):
    """
    most methods in this class are modified from FedML
    """
    __name__ = "FedEMNIST"

    def _preload(self, datadir:Optional[str]=None) -> NoReturn:
        """
        """
        self.datadir = datadir or FED_EMNIST_DATA_DIR

        self.DEFAULT_TRAIN_CLIENTS_NUM = 3400
        self.DEFAULT_TEST_CLIENTS_NUM = 3400
        self.DEFAULT_BATCH_SIZE = 20
        self.DEFAULT_TRAIN_FILE = "fed_emnist_train.h5"
        self.DEFAULT_TEST_FILE = "fed_emnist_test.h5"
        self._IMGAE = "pixels"

        #client id list
        train_file_path = os.path.join(self.datadir, self.DEFAULT_TRAIN_FILE)
        test_file_path = os.path.join(self.datadir, self.DEFAULT_TEST_FILE)
        with h5py.File(train_file_path, "r") as train_h5, h5py.File(test_file_path, "r") as test_h5:
            self._client_ids_train = list(train_h5[self._EXAMPLE].keys())
            self._client_ids_test = list(test_h5[self._EXAMPLE].keys())
            self.n_class = len(np.unique([
                train_h5[self._EXAMPLE][self._client_ids_train[idx]][self._LABEL][0] \
                    for idx in range(self.DEFAULT_TRAIN_CLIENTS_NUM)
            ]))

    def get_dataloader(self,
                       train_bs:int,
                       test_bs:int,
                       client_idx:Optional[int]=None,) -> Tuple[data.DataLoader, data.DataLoader]:
        """
        """
        train_h5 = h5py.File(os.path.join(self.datadir, self.DEFAULT_TRAIN_FILE), "r")
        test_h5 = h5py.File(os.path.join(self.datadir, self.DEFAULT_TEST_FILE), "r")
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
