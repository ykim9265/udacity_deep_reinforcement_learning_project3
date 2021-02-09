
"""
Following code was taken from the DDPG examples provided by the Udacity nanodegree deep reinforcement learning course.

The code has been modified to solve the Reacher environment.

DDPG Examples
-------------
1. https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum
2. https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=256):
        """Initialize parameters and build model.

        Parameters
        ----------
        state_size (int): 
            Dimension of each state

        action_size (int): 
            Dimension of each action

        seed (int): 
            Random seed

        fc1_units (int): 
            Number of nodes in first hidden layer

        fc2_units (int): 
            Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.state_size = state_size
        self.seed = torch.manual_seed(seed)
        self.normalizer = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        
        x = self.normalizer(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Q-Function, Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=512, fc2_units=256, dropout=0.2):
        """Initialize parameters and build model.

        Parameters
        ----------
        state_size (int): 
            Dimension of each state

        action_size (int): 
            Dimension of each action

        seed (int): 
            Random seed

        fcs1_units (int): 
            Number of nodes in the first hidden layer

        fc2_units (int): 
            Number of nodes in the second hidden layer

        dropout: float
            Probability of setting zeros in a given network.
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.normalizer = nn.BatchNorm1d(state_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        
        x = self.normalizer(state)
        x = F.relu(self.fcs1(x))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        return self.fc3(x)
