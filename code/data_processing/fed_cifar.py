"""
federated Cifar10, Cifar100
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


__all__ = ["FedCIFAR", "FedCIFAR100",]


FED_CIFAR_DATA_DIRS = {
    n_class:os.path.join(CACHED_DATA_DIR, f"fed_cifar{n_class}") for n_class in [10, 100,]
}
for n_class in [10, 100,]:
    os.makedirs(FED_CIFAR_DATA_DIRS[n_class], exist_ok=True)


class FedCIFAR(FedVisionDataset):
    """
    most methods in this class are modified from FedML
    """
    __name__ = "FedCIFAR"

    def __init__(self, n_class:int=100, datadir:Optional[str]=None) -> NoReturn:
        """
        """
        self.n_class = n_class
        assert self.n_class in [100,]
        self.datadir = datadir or FED_CIFAR_DATA_DIRS[n_class]

        self.DEFAULT_TRAIN_CLIENTS_NUM = 500
        self.DEFAULT_TEST_CLIENTS_NUM = 100
        self.DEFAULT_BATCH_SIZE = 20
        self.DEFAULT_TRAIN_FILE = "fed_cifar100_train.h5"
        self.DEFAULT_TEST_FILE = "fed_cifar100_test.h5"

        # group name defined by tff in h5 file
        self._EXAMPLE = "examples"
        self._IMGAE = "image"
        self._LABEL = "label"

        #client id list
        train_file_path = os.path.join(self.datadir, self.DEFAULT_TRAIN_FILE)
        test_file_path = os.path.join(self.datadir, self.DEFAULT_TEST_FILE)
        with h5py.File(train_file_path, "r") as train_h5, h5py.File(test_file_path, "r") as test_h5:
            self._client_ids_train = list(train_h5[self._EXAMPLE].keys())
            self._client_ids_test = list(test_h5[self._EXAMPLE].keys())

    def _preload(self) -> NoReturn:
        """
        """
        pass

    def get_dataloader(self,
                       train_bs:int,
                       test_bs:int,
                       client_idx:Optional[int]=None,) -> Tuple[data.DataLoader, data.DataLoader]:
        """
        """
        train_h5 = h5py.File(os.path.join(self.datadir, self.DEFAULT_TRAIN_FILE), "r")
        test_h5 = h5py.File(os.path.join(self.datadir, self.DEFAULT_TEST_FILE), "r")
        train_x, train_y, test_x, test_y = [], [], [], []
        
        # load data in numpy format from h5 file
        if client_idx is None:
            train_x = np.vstack([train_h5[self._EXAMPLE][client_id][self._IMGAE][()] for client_id in self._client_ids_train])
            train_y = np.concatenate([train_h5[self._EXAMPLE][client_id][self._LABEL][()] for client_id in self._client_ids_train])
            test_x = np.vstack([test_h5[self._EXAMPLE][client_id][self._IMGAE][()] for client_id in self._client_ids_test])
            test_y = np.concatenate([test_h5[self._EXAMPLE][client_id][self._LABEL][()] for client_id in self._client_ids_test])
        else:
            client_id_train = self._client_ids_train[client_idx]
            train_x = np.vstack([train_h5[self._EXAMPLE][client_id_train][self._IMGAE][()]])
            train_y = np.concatenate([train_h5[self._EXAMPLE][client_id_train][self._LABEL][()]])
            if client_idx <= len(self._client_ids_test) - 1:
                client_id_test = self._client_ids_test[client_idx]
                test_x = np.vstack([train_h5[self._EXAMPLE][client_id_test][self._IMGAE][()]])
                test_y = np.concatenate([train_h5[self._EXAMPLE][client_id_test][self._LABEL][()]])

        # preprocess
        transform = _data_transforms_fed_cifar(train=True)
        train_x = transform(torch.div(torch.from_numpy(train_x).permute(0,3,1,2), 255.))
        train_y = torch.from_numpy(train_y)
        if len(test_x) != 0:
            transform = _data_transforms_fed_cifar(train=False)
            test_x = transform(torch.div(torch.from_numpy(test_x).permute(0,3,1,2), 255.))
            test_y = torch.from_numpy(test_y)
            pass
        
        # generate dataloader
        train_ds = data.TensorDataset(train_x, train_y)
        train_dl = data.DataLoader(dataset=train_ds,
                                   batch_size=train_bs,
                                   shuffle=True,
                                   drop_last=False,)

        if len(test_x) != 0:
            test_ds = data.TensorDataset(test_x, test_y)
            test_dl = data.DataLoader(dataset=test_ds,
                                      batch_size=test_bs,
                                      shuffle=True,
                                      drop_last=False,)
        else:
            test_dl = None

        train_h5.close()
        test_h5.close()
        return train_dl, test_dl

    def extra_repr_keys(self) -> List[str]:
        """
        """
        return ["n_class",] + super().extra_repr_keys()


class FedCIFAR100(FedCIFAR):
    """
    """
    __name__ = "FedCIFAR100"

    def __init__(self, datadir:Optional[str]=None) -> NoReturn:
        """
        """
        super().__init__(100, datadir)


def _data_transforms_fed_cifar(mean:Optional[Sequence[float]]=None,
                               std:Optional[Sequence[float]]=None,
                               train:bool=True,
                               crop_size:Sequence[int]=(24,24),) -> Callable:
    """
    """
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    if mean is None:
        mean = CIFAR_MEAN
    if std is None:
        std = CIFAR_STD
    if train:
        return transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Normalize(mean=mean, std=std),
        ])
