'''
This script consists of the training classes for the ApiaNet system.

classes:
    - TrainVision: Trains the VisionModule using the visual synthetic dataset and module to inform flight behaviors to attractive and aversive stimuli.
    - TrainGustatory: Trains the GustatoryModule using the gustatory synthetic dataset and module to inform flight behaviors to attractive and aversive stimuli.
    - TrainMotor: Trains the MotorModule using the gustatory synthetic dataset and module to inform flight behaviors to attractive and aversive stimuli.
'''

# Imports
import os
import math
import torch
import random

import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from apianet.dataset.datagen import VisionDataset, GustatoryDataset
from apianet.src.modules import VisionModule, GustatoryModule, MotorModule

class TrainVision(nn.Module):
    def __init__(self, args):
        super(TrainVision, self).__init__()

        # set arguments
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

        # set the model device ("cuda", "MPS", or "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} - note: MPS not supported for vision module yet, see https://github.com/pytorch/pytorch/issues/96056")

    def train(self):
        # Check for existence of vision model before proceeding
        vision_model = os.path.join(self.models_dir, self.vision_model)
        if os.path.exists(vision_model):
            print(f"Vision model already exists at {vision_model}. Overwrite? ((y)/n)")

            # Get user input
            user_input = input().strip().lower()

            if user_input == 'n':
                print("Exiting training.")
                return
            elif user_input not in ('', 'y'):
                print("Invalid input. Exiting training.")
                return
            else:
                print("Continuing training and overwriting existing model.")

        # initialize dataset generator
        image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Create dataset and dataloader.
        dataset = VisionDataset(num_samples=1000, image_transform=image_transform, mask_transform=mask_transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # initialize vision module
        vision = VisionModule().to(self.device)
        vision.train()

        # define optimizer and loss functions
        optimizer = torch.optim.Adam(vision.parameters(), lr=1e-3)
        criterion_color = nn.CrossEntropyLoss()
        criterion_edge = nn.BCEWithLogitsLoss()

        pbar = tqdm(range(self.epoch), desc="Training", unit="epoch")
        for epoch in pbar:
            running_loss = 0.0
            for images, labels, edges in dataloader:
                # Move data to device
                images, labels, edges = images.to(self.device), labels.to(self.device), edges.to(self.device)
                optimizer.zero_grad()
                # Forward pass
                pred_color, pred_edge, _ = vision(images)

                # Resize ground truth edge map to match prediction
                target_edges = torch.nn.functional.interpolate(
                    edges, size=pred_edge.shape[-2:], mode='bilinear', align_corners=False
                )

                # Compute losses
                loss_color = criterion_color(pred_color, labels)
                loss_edge = criterion_edge(pred_edge, target_edges)
                loss = loss_color + loss_edge

                # Backprop and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

            # Compute average loss and update epoch-level progress bar
            epoch_loss = running_loss / len(dataset)
            pbar.set_postfix(loss=epoch_loss)
        
        # save the trained model
        torch.save(vision.state_dict(), vision_model)
        print(f"Model saved to {os.path.join(self.models_dir, self.vision_model)}")

class TrainGustatory(nn.Module):
    def __init__(self, args):
        super(TrainGustatory, self).__init__()

        # set arguments
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

        # set the model device ("cuda", "MPS", or "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def train(self):
        # Check for existence of gustatory model before proceeding
        gust_model = os.path.join(self.models_dir, self.gustatory_model)

        # Confirm user wants to overwrite existing model, default is 'y'
        if os.path.exists(gust_model):
            print(f"Model already exists at {gust_model}. Overwrite? ((y)/n)")

            # Get user input
            user_input = input().strip().lower()

            if user_input == 'n':
                print("Exiting training.")
                return
            elif user_input not in ('', 'y'):
                print("Invalid input. Exiting training.")
                return
            else:
                print("Continuing training and overwriting existing model.")
            
        # Create training dataset and dataloader.
        dataset = GustatoryDataset(num_samples=1000)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Initialize gustatory module
        gustatory = GustatoryModule().to(self.device)
        gustatory.train()

        # define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(gustatory.parameters(), lr=1e-3)

        pbar = tqdm(range(self.epoch), desc="Training Gustatory Module", unit="epoch")
        for epoch in pbar:
            running_loss = 0.0
            for gustatory_vector, label in dataloader:
                # Move data to device
                gustatory_vector = gustatory_vector.to(self.device)
                label = label.to(self.device)

                # Forward + backward + optimize
                optimizer.zero_grad()
                logits = gustatory(gustatory_vector)
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()

                # Accumulate total loss scaled by batch size
                running_loss += loss.item() * gustatory_vector.size(0)

            # Compute average loss across the epoch
            epoch_loss = running_loss / len(dataset)

            # Update tqdm progress bar with average loss
            pbar.set_postfix(loss=epoch_loss)

        # save the trained model
        torch.save(gustatory.state_dict(), gust_model)
        print(f"Model saved to {os.path.join(self.models_dir, self.gustatory_model)}")

class TrainMotor(nn.Module):
    def __init__(self, args):
        super(TrainMotor, self).__init__()

        # set arguments
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

        # set the model device ("cuda", "MPS", or "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # check for existence of gustatory model before proceeding
        self.gust_model = os.path.join(self.models_dir, self.gustatory_model)
        if not os.path.exists(self.gust_model): # prevent further initialization if model not found
            raise FileNotFoundError(f"Gustatory model not found at {self.gust_model}. Please train the gustatory module first.")
        
        # base normalization factor for the gustatory vector
        self.base_norm = np.linalg.norm(np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float32))
        
    def angular_loss(self, pred, tgt):
        """
        Computes the mean angular loss between predicted and target direction vectors.
        Both `pred` and `tgt` should be unit vectors (e.g., [cosθ, sinθ]).

        Args:
            pred (Tensor): Predicted direction vectors, shape [batch_size, 2]
            tgt (Tensor): Target direction vectors, shape [batch_size, 2]

        Returns:
            Tensor: Scalar loss representing mean angular difference (0 = aligned, 2 = opposite)
        """
        # Compute dot product between corresponding vectors in batch
        dot = (pred * tgt).sum(-1).clamp(-1, 1)  # Clamp to handle numerical issues
        # Convert to loss: 1 - cos(θ); ranges from 0 (perfectly aligned) to 2 (opposite)
        return (1 - dot).mean()


    def angle_error_deg(self, pred, tgt):
        """
        Computes the angle error (in degrees) between predicted and target unit vectors.
        Useful as a human-interpretable evaluation metric.

        Args:
            pred (Tensor): Predicted direction vectors, shape [batch_size, 2]
            tgt (Tensor): Target direction vectors, shape [batch_size, 2]

        Returns:
            np.ndarray: Array of angle errors (degrees) for each sample in batch
        """
        # Compute dot product between vectors and clamp to [-1, 1] for safe acos
        dot = (pred * tgt).sum(-1).clamp(-1, 1)
        # Compute angle (radians) and convert to degrees
        angle_rad = torch.acos(dot)  # Arc cosine of dot product gives angle in radians
        return angle_rad.detach().cpu().numpy() * 180 / math.pi
    
    def logistic(self, x):
        """
        A logistic activation function centered at x = 0.5 and scaled by a sharpness factor of 5.0.
        This gives a smooth curve from 0 to 1, useful for smoothly mapping input strengths (like odor norm)
        to velocity magnitudes.

        Args:
            x (float): Input scalar in [0, 1]

        Returns:
            float: Output in (0, 1), steeply increasing around x = 0.5
        """
        return 1 / (1 + math.exp(-5.0 * (x - 0.5)))  # Steep sigmoid around 0.5
    
    def sample_target(self, odor_vec, label, theta0):
        """
        Generates a 3D target output vector [cos(Δθ), sin(Δθ), v] for the motor network.
        Δθ is the angular difference between the desired heading and current heading theta0.
        v is the target speed (in [0, 1]), depending on stimulus strength.

        Args:
            odor_vec (Tensor): A 5D gustatory vector
            label (int): Stimulus label (0 = neutral, 1 = attractive, 2 = aversive)
            theta0 (float): Current heading in radians

        Returns:
            Tensor: A 3D tensor representing direction and velocity target
        """
        # Step 1: Normalize the odor vector strength relative to a predefined base norm
        norm = odor_vec.norm().item() / self.base_norm

        if label == 1:
            # ATTRACTIVE stimulus → Circular scanning pattern
            norm = max(1e-6, min(norm, 1.0))  # Clamp norm for safety

            # Radius decreases with stimulus strength (stronger = tighter)
            # Use an angular offset to simulate curve
            max_offset = math.radians(45)  # wide arc for weak stimulus
            min_offset = math.radians(5)   # tight arc for strong stimulus
            dtheta = max_offset * (1 - norm) + min_offset * norm

            # Turn slightly left (counter-clockwise scan) — could randomize later
            theta_abs = theta0 + dtheta

            # Speed scales *down* with strength
            v = norm
            return torch.tensor([math.cos(theta_abs), math.sin(theta_abs), v], dtype=torch.float32)

        elif label == 2:
            # AVERSIVE stimulus
            # Always turn away (centered at 180°), with small cone of uncertainty
            cone = math.radians(5)
            theta_abs = math.pi + random.gauss(0.0, cone)  # aim opposite current heading
            v = 1.0 * self.logistic(norm)                  # stronger aversion → faster escape

        else:
            # NEUTRAL stimulus — maintain current heading directly
            return torch.tensor([math.cos(theta0), math.sin(theta0), 1.0], dtype=torch.float32)

        # Step 2: Convert absolute target heading into relative heading (Δθ from current direction)
        dtheta = ((theta_abs - theta0 + math.pi) % (2 * math.pi)) - math.pi
        # Step 3: Return unit direction vector [cos(Δθ), sin(Δθ)] and speed scalar
        return torch.tensor([math.cos(dtheta), math.sin(dtheta), v], dtype=torch.float32)
        
    def train(self):
        # Check for existence of motor module before proceeding
        motor_model = os.path.join(self.models_dir, self.motor_model)
        if os.path.exists(motor_model):
            print(f"Motor model already exists at {motor_model}. Overwrite? ((y)/n)")

            # Get user input
            user_input = input().strip().lower()

            if user_input == 'n':
                print("Exiting training.")
                return
            elif user_input not in ('', 'y'):
                print("Invalid input. Exiting training.")
                return
            else:
                print("Continuing training and overwriting existing model.")

        # initialize and load the pre-trained gustatory model
        gustatory = GustatoryModule().to(self.device)
        gustatory.load_state_dict(torch.load(self.gust_model, map_location=self.device, weights_only=True))
        gustatory.eval()

        # initialize gustatory dataset generator
        dataset = GustatoryDataset(num_samples=3000)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # intiialize motor module
        motor = MotorModule().to(self.device)
        motor.train()

        # set up optimizer
        optimizer = torch.optim.Adam(motor.parameters(), lr=1e-3)

        # run main training loop
        pbar = tqdm(range(self.epoch), desc="Training Motor Module")
        for epoch in pbar:
            av_loss = 0.0
            for i, (gustatory_vector, label) in enumerate(dataloader):
                # move data to device
                gustatory_vector = gustatory_vector.to(self.device)
                label = label.to(self.device)

                # forward pass through gustatory module
                with torch.no_grad():
                    gustatory_output = gustatory(gustatory_vector)
                    gustatory_classification = torch.softmax(gustatory_output, dim=1)

                # generate random heading direction
                ang = torch.rand(gustatory_vector.size(0), device=self.device) * 2 * math.pi
                h0 = torch.stack([torch.cos(ang), torch.sin(ang)], 1)

                # prepare motor input
                motor_input = torch.cat([gustatory_output, gustatory_classification, h0], dim=1)

                # motor forward pass
                pred_dir, pred_vel = motor(motor_input)

                # target generation
                target = torch.stack([
                    self.sample_target(o, l.item(), a) 
                    for o, l, a in zip(gustatory_vector, label, ang)
                ]).to(self.device)

                # compute loss
                direction_weight = (label != 0).float() * 1.0 + (label == 0).float() * 0.25
                ang_loss = self.angular_loss(pred_dir, target[:, :2]) * direction_weight
                loss = ang_loss.mean() + nn.functional.mse_loss(pred_vel, target[:, 2])

                # backward + optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update running loss
                av_loss += loss.item()

            # average epoch loss
            av_loss /= len(dataloader)
            pbar.set_postfix(loss=av_loss)
        
        # save the trained model
        torch.save(motor.state_dict(), motor_model)
        print(f"Model saved to {os.path.join(self.models_dir, self.motor_model)}")