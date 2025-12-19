# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the materials for EL2805 - Reinforcement Learning - Exercise Session 3 at KTH, Stockholm.

import numpy as np
import gymnasium as gym
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.tensorboard import SummaryWriter

from DQN_agent import Agent

class MyNetwork(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        assert len(sizes) >= 2

        self.layers = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # no ReLU on output
                x = self.activation(x)
        return x

    def save(self, filename):
        torch.save(self, filename)


class DQNAgent(Agent):
    def __init__(self, network):
        self.network = network

    def forward(self, state: np.ndarray) -> int:
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.network(state)
        return q_values.argmax().item() # to int

if __name__ == '__main__':
    network = torch.load('neural-network-1.pt', weights_only=False)
    DQNAgent(network)
