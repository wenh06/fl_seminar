"""
dataset readers, including Cifar10, Cifar100
"""
from typing import NoReturn, Optional, List, Callable, Tuple

import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100


__all__ = [
    "CIFAR_truncated", "CIFAR10_truncated", "CIFAR100_truncated",
]


class CIFAR_truncated(data.Dataset):
    """
    this class is modified from FedML
    """
    __name__ = "CIFAR10_truncated"

    def __init__(self,
                 root:str,
                 n_class:int=10,
                 dataidxs:Optional[List[int]]=None,
                 train:bool=True,
                 transform:Optional[Callable]=None,
                 target_transform:Optional[Callable]=None,
                 download:bool=False) -> NoReturn:
        """
        """
        self.root = root
        self.n_class = n_class
        assert self.n_class in [10, 100]
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        """
        ds = {10: CIFAR10, 100: CIFAR100}[self.n_class]
        cifar_dataobj = ds(self.root, self.train, self.transform, self.target_transform, self.download)

        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index:np.ndarray) -> NoReturn:
        """
        """
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index:int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class CIFAR10_truncated(CIFAR_truncated):
    """
    """
    __name__ = "CIFAR10_truncated"

    def __init__(self,
                 root:str,
                 dataidxs:Optional[List[int]]=None,
                 train:bool=True,
                 transform:Optional[Callable]=None,
                 target_transform:Optional[Callable]=None,
                 download:bool=False) -> NoReturn:
        """
        """
        super().__init__(root, 10, dataidxs, train, transform, target_transform, download)


class CIFAR100_truncated(CIFAR_truncated):
    """
    """
    __name__ = "CIFAR100_truncated"

    def __init__(self,
                 root:str,
                 dataidxs:Optional[List[int]]=None,
                 train:bool=True,
                 transform:Optional[Callable]=None,
                 target_transform:Optional[Callable]=None,
                 download:bool=False) -> NoReturn:
        """
        """
        super().__init__(root, 100, dataidxs, train, transform, target_transform, download)


class Cutout(object):
    """
    """
    def __init__(self, length:int) -> NoReturn:
        self.length = length

    def __call__(self, img:torch.Tensor) -> torch.Tensor:
        """
        img of shape [..., H, W]
        """
        h, w = img.shape[-2:]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar() -> Tuple[Callable, Callable]:
    """
    """
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        Cutout(16),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, test_transform


def load_cifar_data(datadir:str, n_class:int) -> Tuple[torch.Tensor,...]:
    train_transform, test_transform = _data_transforms_cifar()

    cifar_train_ds = CIFAR_truncated(datadir, n_class, train=True, download=True, transform=train_transform)
    cifar_test_ds = CIFAR_truncated(datadir, n_class, train=False, download=True, transform=test_transform)

    X_train, y_train = cifar_train_ds.data, cifar_train_ds.target
    X_test, y_test = cifar_test_ds.data, cifar_test_ds.target

    return (X_train, y_train, X_test, y_test)
