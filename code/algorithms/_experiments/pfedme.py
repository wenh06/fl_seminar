"""
pFedMe re-implemented in the experiment framework
"""

from copy import deepcopy
from typing import List, NoReturn

from .nodes import Server, Client, SeverConfig, ClientConfig
from .optimizer import get_optimizer


__all__ = [
    "pFedMeServer", "pFedMeClient",
]


class pFedMeServerConfig(SeverConfig):
    """
    """
    __name__ = "pFedMeServerConfig"

    def __init__(self,
                 num_iters:int,
                 num_clients:int,
                 clients_sample_ratio:float,
                 beta:float=1.0,) -> NoReturn:
        """
        """
        super().__init__(
            "pFedMe", num_iters, num_clients, clients_sample_ratio,
            beta=beta,
        )


class pFedMeClientConfig(ClientConfig):
    """
    """
    __name__ = "pFedMeClientConfig"

    def __init__(self,
                 batch_size:int,
                 num_epochs:int,
                 lr:float=0.09,
                 num_steps:int=30,
                 lamda:float=15.0,
                 mu:float=0.001,) -> NoReturn:
        """
        """
        super().__init__(
            "pFedMeClient", "pFedMe", "pFedMe",
            batch_size, num_epochs, lr,
            num_steps=num_steps, lamda=lamda, mu=mu,
        )


class pFedMeServer(Server):
    """
    """
    __name__ = "pFedMeServer"

    @property
    def required_config_fields(self) -> List[str]:
        """
        """
        return ["beta",]
    
    def communicate(self, target:"pFedMeClient") -> NoReturn:
        """
        """
        target._received_messages = {"parameters": deepcopy(self.model.parameters())}

    def update(self) -> NoReturn:
        """
        """
        # store previous parameters
        previous_param = deepcopy(list(self.model.parameters()))

        # sum of received parameters, with self.model.parameters() as its container
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_samples = sum([m["train_samples"] for m in self._received_messages])
        for m in self._received_messages:
            self.add_parameters(m["parameters"], m["train_samples"] / total_samples)

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta) * pre_param.data.detach().clone() + self.beta * param.data

        del pre_param
        self._received_messages = []


class pFedMeClient(Client):
    """
    """
    __name__ = "pFedMeClient"

    @property
    def required_config_fields(self) -> List[str]:
        """
        """
        return ["num_steps", "lamda",]

    def communicate(self, target:"pFedMeServer") -> NoReturn:
        """
        """
        target._received_messages.append(
            {
                "parameters": deepcopy(self.model.parameters()),
                "train_samples": self.config.num_epochs * self.config.num_steps * self.config.batch_size,
            }
        )
        self._received_messages = {}

    def update(self) -> NoReturn:
        """
        """
        self._client_parameters = deepcopy(self._received_messages["parameters"])
        self.train()

    def train(self) -> NoReturn:
        """
        """
        self.model.train()
        for epoch in range(self.config.num_epochs):  # local update
            # self.model.train()
            X, y = self.sample_data()

            # personalized steps
            for i in range(self.config.num_steps):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                updated_parameters, _ = self.optimizer.step(self._client_parameters)

            # update local weight after finding aproximate theta
            for up, cp in zip(updated_parameters, self._client_parameters):
                cp.data.add_(cp.data - up.data, -self.config.lamda * self.config.lr)

            # update local model
            self.set_parameters(self._client_parameters)

    def evaluate(self) -> NoReturn:
        """
        """
        self.model.eval()
