import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_hidden = 128, fc2_hidden = 128, fc3_hidden=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.featLayer = nn.Sequential(
            nn.Linear(state_size, fc1_hidden),
            nn.ReLU(),
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU()
        )
        
        self.valueStream = nn.Sequential(
            nn.Linear(fc2_hidden, fc3_hidden),
            nn.ReLU(),
            nn.Linear(fc3_hidden, 1)
        )
        
        self.advantageStream = nn.Sequential(
            nn.Linear(fc2_hidden, fc3_hidden),
            nn.ReLU(),
            nn.Linear(fc3_hidden, action_size)
        )
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        features = self.featLayer(state)
        values = self.valueStream(features)
        advantages = self.advantageStream(features)
        qvals = values + (advantages - advantages.mean())
        return qvals
