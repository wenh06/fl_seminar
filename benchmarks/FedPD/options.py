import argparse
from typing import Union


def str2bool(v: Union[str, bool]) -> bool:
    """
    References
    ----------
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        b = v
    elif v.lower() in ("yes", "true", "t", "y", "1"):
        b = True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        b = False
    else:
        raise ValueError("Boolean value expected.")
    return b


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of rounds of training"
    )
    parser.add_argument(
        "--freq_in", type=int, default=10, help="number of rounds of local update"
    )
    parser.add_argument(
        "--freq_out", type=int, default=10, help="number of rounds of training"
    )
    parser.add_argument("--num_users", type=int, default=100, help="number of users: K")
    parser.add_argument(
        "--frac", type=float, default=0.1, help="the fraction of clients: C"
    )
    parser.add_argument(
        "--local_ep", type=int, default=10, help="the number of local epochs: E"
    )
    parser.add_argument("--local_bs", type=int, default=10, help="local batch size: B")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--mu", type=float, default=10, help="learning rate")
    parser.add_argument(
        "--momentum", type=float, default=0.5, help="SGD momentum (default: 0.5)"
    )

    # model arguments
    parser.add_argument("--model", type=str, default="mlp", help="model name")
    parser.add_argument(
        "--kernel_num", type=int, default=9, help="number of each kind of kernel"
    )
    parser.add_argument(
        "--kernel_sizes",
        type=str,
        default="3,4,5",
        help="comma-separated kernel size to \
                        use for convolution",
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=1,
        help="number \
                        of channels of imgs",
    )
    parser.add_argument(
        "--norm", type=str, default="batch_norm", help="batch_norm, layer_norm, or None"
    )
    parser.add_argument(
        "--num_filters",
        type=int,
        default=32,
        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.",
    )
    parser.add_argument(
        "--max_pool",
        type=str,
        default="True",
        help="Whether use max pooling rather than \
                        strided convolutions",
    )

    # other arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="name \
                        of dataset",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="number \
                        of classes",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="type \
                        of optimizer",
    )
    parser.add_argument(
        "--VR", type=str2bool, default=False, help="using variance reduction"
    )
    parser.add_argument(
        "--iid", type=int, default=1, help="Default set to IID. Set to 0 for non-IID."
    )
    parser.add_argument(
        "--unequal",
        type=int,
        default=0,
        help="whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)",
    )
    parser.add_argument(
        "--stopping_rounds", type=int, default=10, help="rounds of early stopping"
    )
    parser.add_argument("--verbose", type=int, default=1, help="verbose")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    args = parser.parse_args()

    return args
