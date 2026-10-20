import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self): # initialize object instance
        super().__init__() # initialize nn.Module (parent) object

        # create network
        self.network = nn.Sequential(
            nn.Linear(4, 64), # 4 input features --> 64 first layer outputs
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 1 output mu
        )

        # create a standalone trainable parameter std
        self.log_std = nn.Parameter(torch.zeros(1)) # std always positive when exp

    def forward(self, obs):
        mu = self.network(obs)
        std = torch.exp(self.log_std)
        return mu, std

class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        # create network
        self.network = nn.Sequential(
            nn.Linear(4, 64), # 4 input features --> 64 first layer outputs
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 1 output state value V(s)
        )

    def forward(self, obs):
        value = self.network(obs)
        return value
        


