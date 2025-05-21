'''
This script consists of module classes for the ApiaNet system.

classes:
    - VisionModule: A convolutional neural network (CNN) for processing visual input.
        - Color head: Classifies color information.
        - Edge head: Produces an edge map from the visual input.
        - Edge Encoder: Compresses the edge map into a latent feature vector.
    - GustatoryModule: A simple feedforward neural network to classify gustatory (taste) input vectors.
    - MotorModule: A motor control module that takes an 8D input vector and outputs a normalized 2D movement direction vector and a velocity scalar.
        - Direction head: Predicts a 2D direction vector.
        - Velocity head: Predicts a scalar velocity.
'''
# Imports
import torch

import torch.nn as nn

class VisionModule(nn.Module):
    def __init__(self, num_classes=3, latent_dim=16):
        super().__init__()
        # First block: from input to intermediate features.
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 75x75 -> ~37x37
        )

        # Second block: deeper features.
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # ~37x37 -> ~18x18
        )

        # Color head.
        self.color_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

        # Edge head: produces an initial edge map.
        self.edge_head = nn.Sequential(
            nn.Conv2d(64 + 32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 1, kernel_size=1)
        )

        # Edge Encoder: compresses the edge map to a latent feature.
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, 37, 37]
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B, 32, 19, 19]
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [B, 64, 10, 10] approx.
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((8,8)),  # Instead of (4,4)
            nn.Flatten(),                # [B, 64*8*8 = 4096]
            nn.Linear(4096, latent_dim),  # Compress to latent_dim (latent_dim=64)
            nn.ReLU(inplace=True)         # Adding an extra nonlinearity
        )

    def forward(self, x):
        x1 = self.conv1(x)         # [B, 32, ~37, ~37]
        x2 = self.conv2(x1)        # [B, 64, ~18, ~18]

        color_out = self.color_head(x2)

        # Build input for the edge head.
        x2_up = nn.functional.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        edge_input = torch.cat([x1, x2_up], dim=1)
        # Get the edge map.
        pred_edge = self.edge_head(edge_input)  # [B, 1, ~74, ~74]

        # Encode the edge map into a latent vector.
        latent_edge = self.edge_encoder(pred_edge)  # [B, latent_dim]

        return color_out, pred_edge, latent_edge
    
class GustatoryModule(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, num_classes=3):
        """
        A dense network that classifies a 5D gustatory signal:
            0: Neutral, 1: Attractive, 2: Aversive.
        """
        super(GustatoryModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.out(x)
        return logits

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