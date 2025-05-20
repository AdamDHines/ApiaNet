'''
This script consists of module classes for the ApiaNet system.

classes:
    - GustatoryModule: A simple feedforward neural network to classify gustatory (taste) input vectors.
    - MotorModule: A motor control module that takes an 8D input vector and outputs a normalized 2D movement direction vector and a velocity scalar.
'''
# Imports
import torch

import torch.nn as nn

class GustatoryModule(nn.Module):
    """
    A simple feedforward neural network to classify gustatory (taste) input vectors.
    Input:  5D gustatory vector (e.g., derived from chemical stimulus)
    Output: 3-class logits (for neutral, attractive, aversive)
    """
    def __init__(self, input_dim=5, hidden_dim=32, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # Second hidden layer
        self.out = nn.Linear(hidden_dim, num_classes) # Output layer for classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))     # Apply ReLU after first layer
        x = torch.relu(self.fc2(x))     # Apply ReLU after second layer
        return self.out(x)              # Output raw logits for classification
    
class MotorModule(nn.Module):
    """
    A motor control module that takes an 8D input vector and outputs:
        - a normalized 2D movement direction vector [cosθ, sinθ]
        - a velocity scalar in [0, 1] (via sigmoid)
    """
    def __init__(self):
        super().__init__()
        # Shared processing backbone: two fully connected layers with LayerNorm and ReLU
        self.back = nn.Sequential(
            nn.Linear(8, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU()
        )
        # Separate output heads:
        self.dir = nn.Linear(64, 2)     # Predict 2D direction vector
        self.v   = nn.Linear(64, 1)     # Predict scalar velocity

    def forward(self, x):
        h = self.back(x)                                    # Shared encoding
        d = nn.functional.normalize(self.dir(h), dim=-1, eps=1e-6)  # Normalize direction to unit vector
        v = torch.sigmoid(self.v(h)).squeeze(-1)            # Sigmoid velocity between 0 and 1
        return d, v