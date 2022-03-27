"""
"""

from typing import NoReturn

from misc import experiment_indicator

from data_processing.fed_synthetic import FedSynthetic
from data_processing.fedprox_femnist import FedProxFEMNIST

from algorithms.pfedme import (
    pFedMeClient,
    pFedMeServer,
    pFedMeClientConfig,
    pFedMeServerConfig,
)
from algorithms.fedopt import (
    FedOptClient,
    FedOptServer,
    FedOptClientConfig,
    FedOptServerConfig,
)
from algorithms._experiments.fedprox import (
    FedProxClient,
    FedProxServer,
    FedProxClientConfig,
    FedProxServerConfig,
)
from algorithms._experiments.feddr import (
    FedDRClient,
    FedDRServer,
    FedDRClientConfig,
    FedDRServerConfig,
)


@experiment_indicator("pFedMe")
def test_pfedme() -> NoReturn:
    """ """
    print("Using dataset FedSynthetic")
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = pFedMeServerConfig(10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.7)
    client_config = pFedMeClientConfig(dataset.DEFAULT_BATCH_SIZE, 20)
    s = pFedMeServer(model, dataset, server_config, client_config)
    s.train_centralized()
    s.train_federated()
    del dataset, model, s

    print("Using dataset FedProxFemnist")
    dataset = FedProxFEMNIST()
    model = dataset.candidate_models["cnn_femmist_tiny"]
    server_config = pFedMeServerConfig(10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.2)
    client_config = pFedMeClientConfig(dataset.DEFAULT_BATCH_SIZE, 20)
    s = pFedMeServer(model, dataset, server_config, client_config)
    s.train_centralized()
    s.train_federated()
    del dataset, model, s


@experiment_indicator("FedProx")
def test_fedprox() -> NoReturn:
    """ """
    print("Using dataset FedSynthetic")
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = FedProxServerConfig(10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.7)
    client_config = FedProxClientConfig(dataset.DEFAULT_BATCH_SIZE, 30)
    s = FedProxServer(model, dataset, server_config, client_config)
    s.train_centralized()
    s.train_federated()
    del dataset, model, s

    print("Using dataset FedProxFemnist")
    dataset = FedProxFEMNIST()
    model = dataset.candidate_models["cnn_femmist_tiny"]
    server_config = FedProxServerConfig(10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.2)
    client_config = FedProxClientConfig(dataset.DEFAULT_BATCH_SIZE, 30)
    s = FedProxServer(model, dataset, server_config, client_config)
    s.train_centralized()
    s.train_federated()
    del dataset, model, s


@experiment_indicator("FedDR")
def test_feddr() -> NoReturn:
    """ """
    print("Using dataset FedSynthetic")
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = FedDRServerConfig(10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.7)
    client_config = FedDRClientConfig(dataset.DEFAULT_BATCH_SIZE, 30)
    s = FedDRServer(model, dataset, server_config, client_config)
    s.train_centralized()
    s.train_federated()
    del dataset, model, s

    print("Using dataset FedProxFemnist")
    dataset = FedProxFEMNIST()
    model = dataset.candidate_models["cnn_femmist_tiny"]
    server_config = FedDRServerConfig(10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.2)
    client_config = FedDRClientConfig(dataset.DEFAULT_BATCH_SIZE, 30)
    s = FedDRServer(model, dataset, server_config, client_config)
    s.train_centralized()
    s.train_federated()
    del dataset, model, s


if __name__ == "__main__":
    test_pfedme()
    # test_fedprox()
    # test_feddr()
