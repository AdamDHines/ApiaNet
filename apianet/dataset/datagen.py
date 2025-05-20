'''
This script consists of classes and functions to generate synthetic data for training the ApiaNet system.

classes:
    - GustatoryDataset: Generates neutral, attractive, and aversive gustatory signals to train the gustatory and motor systems of ApiaNet.
'''


# Imports
import torch
import random

import numpy as np

from torch.utils.data import Dataset

class GustatoryDataset(Dataset):
    """
    PyTorch dataset simulating a gustatory (taste-related) environment
    Returns: 
        - gustatory_vector (shape: [5], dtype: float32)
        - label (int): 
            0 = neutral (no overlap with stimulus center),
            1 = attractive (centered overlap, attractive),
            2 = aversive (centered overlap, aversive)
    """
    def __init__(self, num_samples, centre_prob=0.7):
        self.N = num_samples                   # Number of samples in the dataset
        self.centre_prob = centre_prob         # Probability that a patch is sampled from near the arena center

        # Arena and patch geometry
        self.img_size = 450                    # Size of the arena (square image)
        self.patch_size = 75                   # Size of the patch representing honeybee's visual field
        self.small_radius = self.patch_size / 2 # Radius defining the "central" area in the arena
        self.center = (self.img_size // 2, self.img_size // 2)  # Center coordinates of the arena

        self.min_overlap = 0.2                 # Minimum required overlap with center to trigger gustatory signal

        # Baseline gustatory stimulus vector (used for attractive stimuli)
        self.base_gust = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float32)

        # Standard deviation for sampling near the center
        self.sigma = 50                        # Standard deviation of Gaussian sampling around the center

    def __len__(self):
        return self.N

    def __getitem__(self, _):
        # Step 1: Create a circular binary mask marking the "central" area of the arena
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        yy, xx = np.ogrid[:self.img_size, :self.img_size]
        dist = np.sqrt((xx - self.center[0])**2 + (yy - self.center[1])**2)
        mask[dist <= self.small_radius] = 1.0  # Set mask to 1.0 inside the central circle

        # Step 2: Randomly choose whether this patch will be attractive (1) or aversive (2)
        centre_type = random.choice([1, 2])

        # Step 3: Choose the patch center
        if random.random() < self.centre_prob:
            # With probability `centre_prob`, sample near the center using Gaussian distribution
            while True:
                cx = random.gauss(self.center[0], self.sigma)
                cy = random.gauss(self.center[1], self.sigma)
                # Ensure patch doesn't go out of bounds
                if self.patch_size/2 <= cx <= self.img_size - self.patch_size/2 and \
                   self.patch_size/2 <= cy <= self.img_size - self.patch_size/2:
                    break
        else:
            # Otherwise, sample uniformly anywhere within the bounds
            cx = random.uniform(self.patch_size/2, self.img_size - self.patch_size/2)
            cy = random.uniform(self.patch_size/2, self.img_size - self.patch_size/2)

        # Step 4: Extract the rectangular patch and calculate its overlap with the center mask
        l = int(cx - self.patch_size / 2)
        u = int(cy - self.patch_size / 2)
        overlap = mask[u:u+self.patch_size, l:l+self.patch_size].mean()  # Mean value gives overlap percentage

        # Step 5: Generate gustatory vector and label based on overlap and center type
        if overlap >= self.min_overlap:
            if centre_type == 1:
                # Attractive stimulus: use base vector scaled by overlap
                odor = self.base_gust * overlap
            else:
                # Aversive stimulus: inverted base vector scaled by overlap
                odor = -self.base_gust * overlap
            lbl = centre_type
        else:
            # If there's insufficient overlap, treat it as neutral (no stimulus)
            odor = np.zeros(5, dtype=np.float32)
            lbl = 0

        # Step 6: Return as PyTorch tensor and label
        return torch.from_numpy(odor), lbl