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


__all__ = ["FedCIFAR", "FedCIFAR100",]


FED_CIFAR_DATA_DIRS = {
    n_class:os.path.join(CACHED_DATA_DIR, f"fed_cifar{n_class}") for n_class in [10, 100,]
}
for n_class in [10, 100,]:
    os.makedirs(FED_CIFAR_DATA_DIRS[n_class], exist_ok=True)




class FedCIFAR(object):
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
            train_y = np.vstack([train_h5[self._EXAMPLE][client_id][self._LABEL][()] for client_id in self._client_ids_train]).squeeze()
            test_x = np.vstack([test_h5[self._EXAMPLE][client_id][self._IMGAE][()] for client_id in self._client_ids_test])
            test_y = np.vstack([test_h5[self._EXAMPLE][client_id][self._LABEL][()] for client_id in self._client_ids_test]).squeeze()
        else:
            client_id_train = self._client_ids_train[client_idx]
            train_x = np.vstack([train_h5[self._EXAMPLE][client_id_train][self._IMGAE][()]])
            train_y = np.vstack([train_h5[self._EXAMPLE][client_id_train][self._LABEL][()]]).squeeze()
            if client_idx <= len(client_ids_test) - 1:
                client_id_test = self._client_ids_test[client_idx]
                test_x = np.vstack([train_h5[self._EXAMPLE][client_id_test][self._IMGAE][()]])
                test_y = np.vstack([train_h5[self._EXAMPLE][client_id_test][self._LABEL][()]]).squeeze()

        # preprocess
        transform = _data_transforms_fed_cifar(train=True)
        train_x = transform(torch.from_numpy(train_x))
        train_y = torch.from_numpy(train_y)
        if len(test_x) != 0:
            transform = _data_transforms_fed_cifar(train=False)
            test_x = transform(torch.from_numpy(test_x))
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

    def load_partition_data_distributed_federated(self, process_id:int, batch_size:Optional[int]=None) -> tuple:
        """
        """
        _batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        if process_id == 0:
            # get global dataset
            train_data_global, test_data_global = self.get_dataloader(_batch_size, _batch_size)
            train_data_num = len(train_data_global.dataset)
            test_data_num = len(test_data_global.dataset)
            train_data_local = None
            test_data_local = None
            local_data_num = 0
        else:
            # get local dataset
            train_data_local, test_data_local = get_dataloader(_batch_size, _batch_size, process_id - 1)
            train_data_num = local_data_num = len(train_data_local.dataset)
            train_data_global = None
            test_data_global = None
        retval = (
            self.DEFAULT_TRAIN_CLIENTS_NUM,
            train_data_num, train_data_global, test_data_global,
            local_data_num, train_data_local, test_data_local,
            self.n_class,
        )
        return retval

    def load_partition_data_federated(self, batch_size:Optional[int]=0) -> tuple:
        """
        """
        _batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        # get local dataset
        data_local_num_dict = dict()
        train_data_local_dict = dict()
        test_data_local_dict = dict()
        
        for client_idx in range(self.DEFAULT_TRAIN_CLIENTS_NUM):
            train_data_local, test_data_local = self.get_dataloader(_batch_size, _batch_size, client_idx)
            local_data_num = len(train_data_local.dataset)
            data_local_num_dict[client_idx] = local_data_num
            train_data_local_dict[client_idx] = train_data_local
            test_data_local_dict[client_idx] = test_data_local
        
        # global dataset
        train_data_global = \
            data.DataLoader(
                data.ConcatDataset(
                    list(dl.dataset for dl in list(train_data_local_dict.values()))
                ),
                batch_size=_batch_size,
                shuffle=True,
            )
        train_data_num = len(train_data_global.dataset)
        
        test_data_global = \
            data.DataLoader(
                data.ConcatDataset(
                    list(dl.dataset for dl in list(test_data_local_dict.values()) if dl is not None)
                ),
                batch_size=_batch_size,
                shuffle=True,
            )
        test_data_num = len(test_data_global.dataset)

        retval = (
            self.DEFAULT_TRAIN_CLIENTS_NUM,
            train_data_num, test_data_num,
            train_data_global, test_data_global,
            data_local_num_dict, train_data_local_dict, test_data_local_dict,
            self.n_class,
        )

        return retval

    def __repr__(self) -> str:
        return default_class_repr(self)

    def __str__(self) -> str:
        return repr(self)

    def extra_repr_keys(self) -> List[str]:
        """
        """
        return ["n_class", "datadir",]


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
            transforms.ToTensor(),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(crop_size),
            transforms.Normalize(mean=mean, std=std),
        ])
