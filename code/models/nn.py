"""
simple neural network models
"""

from typing import NoReturn

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange


class MLP(nn.Sequential):
    """
    modified from FedPD/models.py
    """

    def __init__(self, dim_in:int, dim_hidden:int, dim_out:int) -> NoReturn:
        """
        """
        super().__init__()
        self.add_module("flatten", Rearrange("b c h w -> b (c h w)"))
        self.add_module("layer_input", nn.Linear(dim_in, dim_hidden))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("dropout", nn.Dropout(p=0.2, inplace=True))
        self.add_module("layer_hidden", nn.Linear(dim_hidden, dim_out))


class CNNMnist(nn.Sequential):
    """
    modified from FedPD/models.py

    input: (batch_size, 1, 28, 28)
    """

    def __init__(self, num_classes:int) -> NoReturn:
        """
        """
        super().__init__()
        self.add_module("conv1", nn.Conv2d(1, 10, kernel_size=5))
        self.add_module("mp1", nn.MaxPool2d(2))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(10, 20, kernel_size=5))
        self.add_module("drop1", nn.Dropout2d(p=0.2, inplace=True))
        self.add_module("mp2", nn.MaxPool2d(2))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("flatten", Rearrange("b c h w -> b (c h w)"))
        self.add_module("fc1", nn.Linear(320, 50))
        self.add_module("relu3", nn.ReLU(inplace=True))
        self.add_module("drop2", nn.Dropout(p=0.2, inplace=True))
        self.add_module("fc2", nn.Linear(50, num_classes))


class CNNFEMnist(nn.Sequential):
    """
    modified from FedPD/models.py

    input shape: (batch_size, 1, 28, 28)
    """

    def __init__(self) -> NoReturn:
        """
        """
        super().__init__()
        self.add_module(
            "conv_block1",
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        )
        self.add_module(
            "conv_block2",
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        )
        self.add_module("flatten", Rearrange("b c h w -> b (c h w)"))
        self.add_module(
            "mlp",
            nn.Sequential(
                nn.Linear(7*7*64, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 62),
            )
        )


class CNNCifar(nn.Sequential):
    """
    modified from FedPD/models.py
    
    input shape： (batch_size, 3, 32, 32)
    """
    def __init__(self, num_classes:int) -> NoReturn:
        """
        """
        super().__init__()
        self.add_module(
            "conv_block1",
            nn.Sequential(
                nn.Conv2d(3, 6, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        )
        self.add_module(
            "conv_block2",
            nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        )
        self.add_module("flatten", Rearrange("b c h w -> b (c h w)"))
        self.add_module(
            "mlp",
            nn.Sequential(
                nn.Linear(16 * 5 * 5, 120),
                nn.ReLU(inplace=True),
                nn.Linear(120, 84),
                nn.ReLU(inplace=True),
                nn.Linear(84, num_classes),
            )
        )
