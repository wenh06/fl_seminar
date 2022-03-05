"""
abstract base classes of federated dataset provided by FedML, and more
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import NoReturn, Optional, Union, List, Callable, Tuple, Dict, Sequence

import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from misc import CACHED_DATA_DIR, ReprMixin


__all__ = [
    "FedDataset",
    "FedVisionDataset",
    "FedNLPDataset",
]


class FedDataset(ReprMixin, ABC):
    """
    """
    __name__ = "FedDataset"

    @abstractmethod
    def get_dataloader(self,
                       train_bs:int,
                       test_bs:int,
                       client_idx:Optional[int]=None,) -> Tuple[data.DataLoader, data.DataLoader]:
        """
        """
        raise NotImplementedError

    @abstractmethod
    def _preload(self, datadir:Optional[str]=None) -> NoReturn:
        """
        """
        raise NotImplementedError

    @abstractmethod
    def load_partition_data_distributed(self, process_id:int, batch_size:Optional[int]=None) -> tuple:
        """ get local dataloader at client `process_id` or get global dataloader
        """
        raise NotImplementedError

    @abstractmethod
    def load_partition_data(self, batch_size:Optional[int]=None) -> tuple:
        """ partition data into all local clients
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, probs:torch.Tensor, truths:torch.Tensor) -> Dict[str, float]:
        """ 
        """
        raise NotImplementedError

    def extra_repr_keys(self) -> List[str]:
        """
        """
        return super().extra_repr_keys() + ["datadir",]


class FedVisionDataset(FedDataset, ABC):
    """
    """
    __name__ = "FedVisionDataset"

    def __init__(self, datadir:Optional[Union[Path,str]]=None) -> NoReturn:
        """
        """
        self.datadir = Path(datadir) if datadir is not None else None

        self.DEFAULT_TRAIN_CLIENTS_NUM = None
        self.DEFAULT_TEST_CLIENTS_NUM = None
        self.DEFAULT_BATCH_SIZE = None
        self.DEFAULT_TRAIN_FILE = None
        self.DEFAULT_TEST_FILE = None

        # group name defined by tff in h5 file
        self._EXAMPLE = "examples"
        self._IMGAE = "image"
        self._LABEL = "label"

        self._preload(datadir)

        assert all([
            self.criterion is not None,
            self.datadir is not None,
            self.DEFAULT_TRAIN_CLIENTS_NUM is not None,
            self.DEFAULT_TEST_CLIENTS_NUM is not None,
            self.DEFAULT_BATCH_SIZE is not None,
            self.DEFAULT_TRAIN_FILE is not None,
            self.DEFAULT_TEST_FILE is not None,
        ])

    @abstractmethod
    def get_dataloader(self,
                       train_bs:int,
                       test_bs:int,
                       client_idx:Optional[int]=None,) -> Tuple[data.DataLoader, data.DataLoader]:
        """
        """
        raise NotImplementedError

    @abstractmethod
    def _preload(self, datadir:Optional[str]=None) -> NoReturn:
        """
        """
        raise NotImplementedError

    def load_partition_data_distributed(self, process_id:int, batch_size:Optional[int]=None) -> tuple:
        """ get local dataloader at client `process_id` or get global dataloader
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
            train_data_local, test_data_local = self.get_dataloader(_batch_size, _batch_size, process_id - 1)
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

    def load_partition_data(self, batch_size:Optional[int]=None) -> tuple:
        """ partition data into all local clients
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

    @property
    def n_class(self) -> int:
        return self._n_class


class FedNLPDataset(FedDataset, ABC):
    """
    """
    __name__ = "FedNLPDataset"

    def __init__(self, datadir:Optional[Union[str,Path]]=None) -> NoReturn:
        """
        """
        self.datadir = Path(datadir) if datadir is not None else None

        self.DEFAULT_TRAIN_CLIENTS_NUM = None
        self.DEFAULT_TEST_CLIENTS_NUM = None
        self.DEFAULT_BATCH_SIZE = None
        self.DEFAULT_TRAIN_FILE = None
        self.DEFAULT_TEST_FILE = None

        self._preload()

        assert all([
            self.criterion is not None,
            self.datadir is not None,
            self.DEFAULT_TRAIN_CLIENTS_NUM is not None,
            self.DEFAULT_TEST_CLIENTS_NUM is not None,
            self.DEFAULT_BATCH_SIZE is not None,
            self.DEFAULT_TRAIN_FILE is not None,
            self.DEFAULT_TEST_FILE is not None,
        ])

    @abstractmethod
    def get_dataloader(self,
                       train_bs:int,
                       test_bs:int,
                       client_idx:Optional[int]=None,) -> Tuple[data.DataLoader, data.DataLoader]:
        """
        """
        raise NotImplementedError

    def load_partition_data_distributed(self, process_id:int, batch_size:Optional[int]=None) -> tuple:
        """ get local dataloader at client `process_id` or get global dataloader
        """
        _batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        if process_id == 0:
            # get global dataset
            train_data_global, test_data_global = self.get_dataloader(batch_size, batch_size)
            train_data_num = len(train_data_global.dataset)
            test_data_num = len(test_data_global.dataset)
            train_data_local = None
            test_data_local = None
            local_data_num = 0
        else:
            # get local dataset
            train_data_local, test_data_local = get_dataloader(batch_size, batch_size, process_id - 1)
            train_data_num = local_data_num = len(train_data_local.dataset)
            train_data_global = None
            test_data_global = None
            
        VOCAB_LEN = len(self.get_word_dict()) + 1

        retval = (
            self.DEFAULT_TRAIN_CLIENTS_NUM,
            train_data_num, train_data_global, test_data_global,
            local_data_num, train_data_local, test_data_local,
            VOCAB_LEN,
        )

        return retval

    def load_partition_data(self, batch_size:Optional[int]=None) -> tuple:
        """ partition data into all local clients
        """
        _batch_size = batch_size or self.DEFAULT_BATCH_SIZE

        # get local dataset
        data_local_num_dict = dict()
        train_data_local_dict = dict()
        test_data_local_dict = dict()

        for client_idx in range(self.DEFAULT_TRAIN_CLIENTS_NUM):
            train_data_local, test_data_local = self.get_dataloader(batch_size, batch_size, client_idx)
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
                batch_size=batch_size,
                shuffle=True,
            )
        train_data_num = len(train_data_global.dataset)

        test_data_global = \
            data.DataLoader(
                data.ConcatDataset(
                    list(dl.dataset for dl in list(test_data_local_dict.values()) if dl is not None)
                ),
                batch_size=batch_size,
                shuffle=True,
            )
        test_data_num = len(test_data_global.dataset)

        VOCAB_LEN = len(self.get_word_dict()) + 1

        retval = (
            self.DEFAULT_TRAIN_CLIENTS_NUM,
            train_data_num, test_data_num, train_data_global, test_data_global,
            data_local_num_dict, train_data_local_dict, test_data_local_dict,
            VOCAB_LEN,
        )

        return retval

    @abstractmethod
    def get_word_dict(self) -> Dict[str,int]:
        """
        """
        raise NotImplementedError
