"""
built-in simple models
"""

from .nn import (
    MLP,
    FedPDMLP,
    CNNMnist,
    CNNFEMnist,
    CNNFEMnist_Tiny,
    CNNCifar,
    RNN_OriginalFedAvg,
    RNN_StackOverFlow,
    ResNet18,
    ResNet10,
)


__all__ = [
    "MLP",
    "FedPDMLP",
    "CNNMnist",
    "CNNFEMnist",
    "CNNFEMnist_Tiny",
    "CNNCifar",
    "RNN_OriginalFedAvg",
    "RNN_StackOverFlow",
    "ResNet18",
    "ResNet10",
]
