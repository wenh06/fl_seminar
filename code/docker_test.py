"""
"""

from functools import wraps
from typing import Callable, Any

from data_processing.fed_synthetic import FedSynthetic
from data_processing.fedprox_femnist import FedProxFEMNIST

from algorithms.pfedme import pFedMeClient, pFedMeServer, pFedMeClientConfig, pFedMeServerConfig
from algorithms._experiments.fedprox import FedProxClient, FedProxServer, FedProxClientConfig, FedProxServerConfig
from algorithms._experiments.feddr import FedDRClient, FedDRServer, FedDRClientConfig, FedDRServerConfig


def experiment_indicator(name:str) -> Callable:
    """
    """
    def decorator(func:Callable) -> Callable:
        @wraps(func)
        def wrapper(*args:Any, **kwargs:Any) -> Any:
            print("\n" + "-" * 100)
            print(f"  Start experiment {name}  ".center(100, "-"))
            print("-" * 100 + "\n")
            func(*args, **kwargs)
            print("\n" + "-" * 100)
            print(f"  End experiment {name}  ".center(100, "-"))
            print("-" * 100 + "\n")
        return wrapper
    return decorator


@experiment_indicator("pFedMe")
def test_pfedme():
    """
    """
    print("Using dataset FedSynthetic")
    dataset = FedSynthetic(1,1,False,30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = pFedMeServerConfig(10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.7)
    client_config = pFedMeClientConfig(dataset.DEFAULT_BATCH_SIZE, 30)
    s = pFedMeServer(model, dataset, server_config, client_config)
    s.train_centralized()
    s.train_federated()
    del dataset, model, s

    print("Using dataset FedProxFemnist")
    dataset = FedProxFEMNIST()
    model = dataset.candidate_models["cnn_femmist_tiny"]
    server_config = pFedMeServerConfig(10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.7)
    client_config = pFedMeClientConfig(dataset.DEFAULT_BATCH_SIZE, 30)
    s = pFedMeServer(model, dataset, server_config, client_config)
    s.train_centralized()
    s.train_federated()
    del dataset, model, s


@experiment_indicator("FedProx")
def test_fedprox():
    """
    """
    print("Using dataset FedSynthetic")
    dataset = FedSynthetic(1,1,False,30)
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
    server_config = FedProxServerConfig(10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.7)
    client_config = FedProxClientConfig(dataset.DEFAULT_BATCH_SIZE, 30)
    s = FedProxServer(model, dataset, server_config, client_config)
    s.train_centralized()
    s.train_federated()
    del dataset, model, s


@experiment_indicator("FedDR")
def test_feddr():
    """
    """
    print("Using dataset FedSynthetic")
    dataset = FedSynthetic(1,1,False,30)
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
    server_config = FedDRServerConfig(10, dataset.DEFAULT_TRAIN_CLIENTS_NUM, 0.7)
    client_config = FedDRClientConfig(dataset.DEFAULT_BATCH_SIZE, 30)
    s = FedDRServer(model, dataset, server_config, client_config)
    s.train_centralized()
    s.train_federated()
    del dataset, model, s


if __name__ == "__main__":
    test_pfedme()
    # test_fedprox()
    # test_feddr()
