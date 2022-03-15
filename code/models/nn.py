"""
simple neural network models
"""

import re
from typing import NoReturn, Optional, Union, Sequence

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, resnet18
import einops
from einops.layers.torch import Rearrange

from .utils import SizeMixin, CLFMixin


__all__ = [
    "MLP", "FedPDMLP",
    "CNNMnist",
    "CNNFEMnist", "CNNFEMnist_Tiny",
    "CNNCifar",
    "RNN_OriginalFedAvg",
    "RNN_StackOverFlow",
    "ResNet18", "ResNet10",
]


class MLP(CLFMixin, SizeMixin, nn.Sequential):
    """
    multi-layer perceptron

    can be used for
    1. logistic regression (for classification) using cross entropy loss (CrossEntropyLoss, BCEWithLogitsLoss, etc)
    2. regression (for regression) using MSE loss
    3. SVM (for classification) using hinge loss (MultiMarginLoss, MultiLabelMarginLoss, etc)
    4. etc.
    """
    __name__ = "MLP"

    def __init__(self, dim_in:int, dim_out:int, dim_hidden:Optional[Union[int,Sequence[int]]]=None) -> NoReturn:
        """
        """
        super().__init__()
        self.add_module("flatten", Rearrange("b c h w -> b (c h w)"))
        dims = []
        if dim_hidden is not None:
            if isinstance(dim_hidden, int):
                dims = [dim_hidden,]
            else:
                dims = list(dim_hidden)
        for i, dim in enumerate(dims):
            self.add_module(f"linear_{i+1}", nn.Linear(dim_in, dim))
            self.add_module(f"relu_{i+1}", nn.ReLU())
            self.add_module(f"dropout_{i+1}", nn.Dropout(p=0.2, inplace=True))
            dim_in = dim
        self.add_module(f"linear_{len(dims)+1}", nn.Linear(dim_in, dim_out))

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        for _ in range(4-input.ndim):
            # allow for input of ndim 2 and 3
            input = input.unsqueeze(1)
        return super().forward(input)


class FedPDMLP(CLFMixin, SizeMixin, nn.Sequential):
    """
    modified from FedPD/models.py
    """
    __name__ = "FedPDMLP"

    def __init__(self, dim_in:int, dim_hidden:int, dim_out:int) -> NoReturn:
        """
        """
        super().__init__()
        self.add_module("flatten", Rearrange("b c h w -> b (c h w)"))
        self.add_module("layer_input", nn.Linear(dim_in, dim_hidden))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("dropout", nn.Dropout(p=0.2, inplace=True))
        self.add_module("layer_hidden", nn.Linear(dim_hidden, dim_out))

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        for _ in range(4-input.ndim):
            # allow for input of ndim 2 and 3
            input = input.unsqueeze(1)
        return super().forward(input)


class CNNMnist(CLFMixin, SizeMixin, nn.Sequential):
    """
    modified from FedPD/models.py

    input: (batch_size, 1, 28, 28)
    """
    __name__ = "CNNMnist"

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


class CNNFEMnist(CLFMixin, SizeMixin, nn.Sequential):
    """
    modified from FedPD/models.py

    input shape: (batch_size, 1, 28, 28)
    """
    __name__ = "CNNFEMnist"

    def __init__(self) -> NoReturn:
        """
        """
        super().__init__()
        in_channels = 1
        for i, out_channels in enumerate([32, 64]):
            self.add_module(
                f"conv_block{i+1}",
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
            in_channels = out_channels
        self.add_module("flatten", Rearrange("b c h w -> b (c h w)"))
        self.add_module(
            "mlp",
            nn.Sequential(
                nn.Linear(7*7*in_channels, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 62),
            )
        )


class CNNFEMnist_Tiny(CLFMixin, SizeMixin, nn.Sequential):
    """
    modified from FedPD/models.py

    input shape: (batch_size, 1, 28, 28)
    """
    __name__ = "CNNFEMnist_Tiny"

    def __init__(self) -> NoReturn:
        """
        """
        super().__init__()
        in_channels = 1
        for i, out_channels in enumerate([16, 32]):
            self.add_module(
                f"conv_block{i+1}",
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
            in_channels = out_channels
        self.add_module("flatten", Rearrange("b c h w -> b (c h w)"))
        self.add_module(
            "mlp",
            nn.Sequential(
                nn.Linear(7*7*in_channels, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 62),
            )
        )


class CNNCifar(CLFMixin, SizeMixin, nn.Sequential):
    """
    modified from FedPD/models.py
    
    input shape： (batch_size, 3, 32, 32)
    """
    __name__ = "CNNCifar"

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


class RNN_OriginalFedAvg(CLFMixin, SizeMixin, nn.Module):
    """Creates a RNN model using LSTM layers for Shakespeare language models (next character prediction task).
      This replicates the model structure in the paper:
      Communication-Efficient Learning of Deep Networks from Decentralized Data
        H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agueray Arcas. AISTATS 2017.
        https://arxiv.org/abs/1602.05629
      This is also recommended model by "Adaptive Federated Optimization. ICML 2020" (https://arxiv.org/pdf/2003.00295.pdf)

    Modified from `FedML`
    """
    __name__ = "RNN_OriginalFedAvg"

    def __init__(self, embedding_dim:int=8, vocab_size:int=90, hidden_size:int=256) -> NoReturn:
        """
        """
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq:Tensor) -> Tensor:
        """
        """
        embeds = self.embeddings(input_seq)
        # Note that the order of mini-batch is random so there is no hidden relationship among batches.
        # So we do not input the previous batch's hidden state,
        # leaving the first hidden state zero `self.lstm(embeds, None)`.
        lstm_out, _ = self.lstm(embeds)
        # use the final hidden state as the next character prediction
        # final_hidden_state = lstm_out[:, -1]
        # output = self.fc(final_hidden_state)
        # For fed_shakespeare
        output = einops.rearrange(
            self.fc(lstm_out),
            "batch length vocab -> batch vocab length"
        )
        return output


class RNN_StackOverFlow(CLFMixin, SizeMixin, nn.Module):
    """Creates a RNN model using LSTM layers for StackOverFlow (next word prediction task).
      This replicates the model structure in the paper:
      "Adaptive Federated Optimization. ICML 2020" (https://arxiv.org/pdf/2003.00295.pdf)
      Table 9
    
    Modified from `FedML`
    """
    __name__ = "RNN_StackOverFlow"

    def __init__(self,
                 vocab_size:int=10000,
                 num_oov_buckets:int=1,
                 embedding_size:int=96,
                 latent_size:int=670,
                 num_layers:int=1) -> NoReturn:
        """
        """
        super().__init__()
        extended_vocab_size = vocab_size + 3 + num_oov_buckets  # For pad/bos/eos/oov.
        self.word_embeddings = nn.Embedding(num_embeddings=extended_vocab_size, embedding_dim=embedding_size,
                                            padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=latent_size, num_layers=num_layers)
        self.fc1 = nn.Linear(latent_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, extended_vocab_size)

    def forward(self, input_seq:Tensor, hidden_state:Optional[Tensor]=None) -> Tensor:
        """
        """
        embeds = self.word_embeddings(input_seq)
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        fc1_output = self.fc1(lstm_out)
        output = einops.rearrange(
            self.fc2(fc1_output),
            "batch length vocab -> batch vocab length"
        )
        return output


class ResNet18(CLFMixin, SizeMixin, ResNet):
    """
    """
    __name__ = "ResNet18"

    def __init__(self, num_classes:int, pretrained:bool=False) -> NoReturn:
        """
        """
        super().__init__(
            BasicBlock, [2, 2, 2, 2],
            num_classes=num_classes,
        )
        if pretrained:
            _model = resnet18(pretrained=True)
            self.load_state_dict(
                {k:v for k,v in _model.state_dict().items() if not re.findall("^fc\.", k)},
                strict=False,
            )
            del _model


class ResNet10(CLFMixin, SizeMixin, ResNet):
    """
    """
    __name__ = "ResNet10"

    def __init__(self, num_classes:int, pretrained:bool=False) -> NoReturn:
        """
        """
        super().__init__(
            BasicBlock, [1, 1, 1, 1],
            num_classes=num_classes,
        )
        if pretrained:
            raise NotImplementedError("ResNet10 has no pretrained model.")
