"""
"""

from pathlib import Path
from collections import OrderedDict
from itertools import repeat
from typing import NoReturn, Optional, Union, List, Callable, Tuple, Dict, Sequence

import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from misc import CACHED_DATA_DIR, default_class_repr
from .fed_dataset import FedNLPDataset


__all__ = ["FedShakespeare",]


FED_SHAKESPEARE_DATA_DIR = CACHED_DATA_DIR / "fed_shakespeare"
FED_SHAKESPEARE_DATA_DIR.mkdir(exist_ok=True)


class FedShakespeare(FedNLPDataset):
    """
    """
    __name__ = "FedShakespeare"

    def _preload(self, datadir:Optional[Union[str,Path]]=None) -> NoReturn:
        """
        """
        self.datadir = Path(datadir or FED_SHAKESPEARE_DATA_DIR)

        self.SEQUENCE_LENGTH = 80  # from McMahan et al AISTATS 2017
        # Vocabulary re-used from the Federated Learning for Text Generation tutorial.
        # https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation
        self.CHAR_VOCAB = list(
            'dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r'
        )
        self._pad = "<pad>"
        self._bos = "<bos>"
        self._eos = "<eos>"
        self._oov = "<oov>"

        self._words = [self._pad] + self.CHAR_VOCAB + [self._bos] + [self._eos]
        self.word_dict = OrderedDict()
        for i, w in enumerate(self._words):
            self.word_dict[w] = i

        self.DEFAULT_TRAIN_CLIENTS_NUM = 715
        self.DEFAULT_TEST_CLIENTS_NUM = 715
        self.DEFAULT_BATCH_SIZE = 4
        self.DEFAULT_TRAIN_FILE = "shakespeare_train.h5"
        self.DEFAULT_TEST_FILE = "shakespeare_test.h5"

        # group name defined by tff in h5 file
        self._EXAMPLE = "examples"
        self._SNIPPETS = "snippets"

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        train_file_path = self.datadir / self.DEFAULT_TRAIN_FILE
        test_file_path = self.datadir / self.DEFAULT_TEST_FILE
        with h5py.File(str(train_file_path), "r") as train_h5, h5py.File(str(test_file_path), "r") as test_h5:
            self._client_ids_train = list(train_h5[self._EXAMPLE].keys())
            self._client_ids_test = list(test_h5[self._EXAMPLE].keys())

    def get_dataloader(self,
                       train_bs:int,
                       test_bs:int,
                       client_idx:Optional[int]=None,) -> Tuple[data.DataLoader, data.DataLoader]:
        """
        """
        train_h5 = h5py.File(str(self.datadir / self.DEFAULT_TRAIN_FILE), "r")
        test_h5 = h5py.File(str(self.datadir / self.DEFAULT_TEST_FILE), "r")
        train_ds = []
        test_ds = []

        # load data
        if client_idx is None:
            # get ids of all clients
            train_ids = self._client_ids_train
            test_ids = self._client_ids_test
        else:
            # get ids of single client
            train_ids = [self._client_ids_train[client_idx]]
            test_ids = [self._client_ids_test[client_idx]]

        for client_id in train_ids:
            raw_train = train_h5[self._EXAMPLE][client_id][self._SNIPPETS][()]
            raw_train = [x.decode("utf8") for x in raw_train]
            train_ds.extend(self.preprocess(raw_train))
        for client_id in test_ids:
            raw_test = test_h5[self._EXAMPLE][client_id][self._SNIPPETS][()]
            raw_test = [x.decode("utf8") for x in raw_test]
            test_ds.extend(self.preprocess(raw_test))

        # split data
        train_x, train_y = FedShakespeare._split_target(train_ds)
        test_x, test_y = FedShakespeare._split_target(test_ds)
        train_ds = data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
        test_ds = data.TensorDataset(torch.tensor(test_x), torch.tensor(test_y))
        train_dl = data.DataLoader(
            dataset=train_ds,
            batch_size=train_bs,
            shuffle=True,
            drop_last=False,
        )
        test_dl = data.DataLoader(
            dataset=test_ds,
            batch_size=test_bs,
            shuffle=True,
            drop_last=False,
        )

        train_h5.close()
        test_h5.close()
        return train_dl, test_dl

    @staticmethod
    def _split_target(sequence_batch: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Split a N + 1 sequence into shifted-by-1 sequences for input and output."""
        sequence_batch = np.asarray(sequence_batch)
        input_text = sequence_batch[...,:-1]
        target_text = sequence_batch[...,1:]
        return (input_text, target_text)

    def preprocess(self, sentences:Sequence[str], max_seq_len:Optional[int]=None) -> List[List[int]]:
        """
        """
        sequences = []
        if max_seq_len is None:
            max_seq_len = self.SEQUENCE_LENGTH

        def to_ids(sentence:str, num_oov_buckets:int=1) -> Tuple[List[int]]:
            """
            map list of sentence to list of [idx..] and pad to max_seq_len + 1
            Args:
                num_oov_buckets : The number of out of vocabulary buckets.
                max_seq_len: Integer determining shape of padded batches.
            """
            tokens = [self.char_to_id(c) for c in sentence]
            tokens = [self.char_to_id(self._bos)] + tokens + [self.char_to_id(self._eos)]
            if len(tokens) % (max_seq_len + 1) != 0:
                pad_length = (-len(tokens)) % (max_seq_len + 1)
                tokens += list(repeat(self.char_to_id(self._pad), pad_length))
            return (tokens[i:i + max_seq_len + 1]
                    for i in range(0, len(tokens), max_seq_len + 1))

        for sen in sentences:
            sequences.extend(to_ids(sen))
        return sequences

    def id_to_word(self, idx:int) -> str:
        return self.words[idx]

    def char_to_id(self, char:str) -> int:
        return self.word_dict.get(char, len(self.word_dict))

    @property
    def words(self) -> List[str]:
        return self._words

    def get_word_dict(self) -> Dict[str,int]:
        return self.word_dict

    def evaluate(self, preds:torch.Tensor, truths:torch.Tensor) -> Dict[str, float]:
        """
        """
        pass
