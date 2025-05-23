'''
This script consists of the evaluation classes for the ApiaNet system.

classes:
    - EvalVision: Class for evaluating the vision module of ApiaNet.
    - EvalGustatory: Class for evaluating the gustatory module of ApiaNet. 
    - EvalMotor: Class for evaluating the motor module of ApiaNet.
'''

# Imports
import os
import math
import torch
import random

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import pearsonr
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from apianet.dataset.datagen import VisionDataset, GustatoryDataset
from apianet.src.modules import VisionModule, GustatoryModule, MotorModule

class EvalVision:
    '''
    Class for evaluating the vision module of ApiaNet.
    '''
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision = VisionModule().to(self.device)
        self.vision.load_state_dict(torch.load(os.path.join(args.models_dir, args.vision_model), weights_only=True))
        self.vision.eval()
        
        # Define the dataset and dataloader
        image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.dataset = VisionDataset(num_samples=10, image_transform=image_transform, mask_transform=mask_transform, eval=True)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

    def eval(self):
        correct = 0
        total = 0
        edge_sim_total = 0.0
        plot_samples = []  # (image, ground truth label, predicted label, color probs, predicted edge, gt edge)

        with torch.no_grad():
            for images, labels, gt_edges in tqdm(self.dataloader, desc="Validating"):
                images, labels, gt_edges = images.to(self.device), labels.to(self.device), gt_edges.to(self.device)

                pred_color, pred_edge, _ = self.vision(images)
                predicted_labels = pred_color.argmax(dim=1)
                correct += (predicted_labels == labels).sum().item()
                total += labels.size(0)

                # Cosine similarity between predicted edge and ground truth edge
                # Resize predicted edge to match ground truth shape
                if pred_edge.shape[-2:] != gt_edges.shape[-2:]:
                    pred_edge = torch.nn.functional.interpolate(
                        pred_edge, size=gt_edges.shape[-2:], mode='bilinear', align_corners=False
                    )
                pred_edge_flat = pred_edge.view(pred_edge.size(0), -1)
                gt_edge_flat = gt_edges.view(gt_edges.size(0), -1)
                edge_similarity = torch.nn.functional.cosine_similarity(pred_edge_flat, gt_edge_flat, dim=1)  # shape: [batch]
                edge_sim_total += edge_similarity.sum().item()

                # Collect samples for plotting
                color_probs = torch.softmax(pred_color, dim=1).cpu()
                edge_pred = torch.sigmoid(pred_edge).cpu()
                for i in range(min(len(images), 3 - len(plot_samples))):
                    plot_samples.append((
                        images[i].cpu(),
                        labels[i].cpu(),
                        predicted_labels[i].cpu(),
                        color_probs[i],
                        edge_pred[i],
                        gt_edges[i].cpu()
                    ))
                    if len(plot_samples) >= 3:
                        break

        acc = correct / total * 100
        edge_sim_avg = edge_sim_total / total * 100  # convert to percentage

        print(f"\nValidation Accuracy over {total} samples: {acc:.2f}%")
        print(f"Average Edge Cosine Similarity: {edge_sim_avg:.2f}%")

        # ---------------------------
        # Plotting
        # ---------------------------
        label_map = {0: "Other", 1: "Green", 2: "Blue"}
        num_samples_to_plot = len(plot_samples)
        plt.figure(figsize=(15, num_samples_to_plot * 3))

        for idx, (img_tensor, true_label, pred_label, color_probs, edge_pred, gt_edge) in enumerate(plot_samples):
            img_np = np.transpose(img_tensor.numpy(), (1, 2, 0))

            plt.subplot(num_samples_to_plot, 4, idx * 4 + 1)
            plt.imshow(img_np)
            plt.title(f"Patch\nGT: {label_map.get(true_label.item(), 'N/A')}\nPred: {label_map.get(pred_label.item(), 'N/A')}")
            plt.axis("off")

            plt.subplot(num_samples_to_plot, 4, idx * 4 + 2)
            probs = color_probs.numpy()
            classes = np.arange(3)
            plt.bar(classes, probs, color=["gray", "green", "blue"])
            plt.xticks(classes, [label_map[c] for c in classes])
            plt.ylabel("Probability")
            plt.ylim(0, 1)
            plt.title("Color Module Output")

            plt.subplot(num_samples_to_plot, 4, idx * 4 + 3)
            edge_np = edge_pred.squeeze().numpy()
            plt.imshow(edge_np, cmap="gray")
            plt.title("Predicted Edge")
            plt.axis("off")

            plt.subplot(num_samples_to_plot, 4, idx * 4 + 4)
            gt_np = gt_edge.squeeze().numpy()
            plt.imshow(gt_np, cmap="gray")
            plt.title("Ground Truth Edge")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

class EvalGustatory:
    '''
    Class for evaluating the gustatory module of ApiaNet.
    '''
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gustatory = GustatoryModule().to(self.device)
        self.gustatory.load_state_dict(torch.load(os.path.join(args.models_dir, args.gustatory_model), weights_only=True))
        self.gustatory.eval()
        
        # Define the dataset and dataloader
        self.dataset = GustatoryDataset(num_samples=1000)
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=False)


    def eval(self):
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for odor_vector, label in tqdm(self.dataloader, desc="Evaluating Gustatory Module"):
                odor_vector, label = odor_vector.to(self.device), label.to(self.device)
                logits = self.gustatory(odor_vector)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(label.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Compute confusion matrix.
        cm = confusion_matrix(all_labels, all_preds)

        # Plot confusion matrix.
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Gustatory Module – Confusion Matrix")
        plt.colorbar()
        class_names = ['Neutral', 'Attractive', 'Aversive']
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

class EvalMotor:
    '''
    Class for evaluating the motor module of ApiaNet, driven by the gustatory module.
    '''
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Constants
        self.IMG_SIZE = 450
        self.SMALL_RADIUS = 37.5
        self.PATCH_SIZE = 75
        self.CENTER = (self.IMG_SIZE // 2, self.IMG_SIZE // 2)

        self.BASE_APP = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float32)
        self.BASE_AVR = -self.BASE_APP
        self.BASE_NORM = np.linalg.norm(self.BASE_APP)
        self.V_MIN = 0.0
        self.V_MAX = 1.0
        self.SPEED_K = 5.0
        self.MIN_OVERLAP = 0.2

        # Load frozen gustatory module
        gust_path = os.path.join(args.models_dir, args.gustatory_model)
        self.gustatory = GustatoryModule().to(self.device)
        state = torch.load(gust_path, map_location=self.device, weights_only=True)
        self.gustatory.load_state_dict(state)
        self.gustatory.eval()
        for p in self.gustatory.parameters():
            p.requires_grad_(False)

        # Load trained motor module
        motor_path = os.path.join(args.models_dir, args.motor_model)
        self.motor = MotorModule().to(self.device)
        self.motor.load_state_dict(torch.load(motor_path, map_location=self.device, weights_only=True))
        self.motor.eval()

        # Dataset and dataloader
        self.dataset = GustatoryDataset(num_samples=1000)
        self.dataloader = DataLoader(self.dataset, batch_size=128, shuffle=False)

    def logistic(self, x):
        return 1 / (1 + math.exp(-self.SPEED_K * (x - 0.5)))

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
        norm = odor_vec.norm().item() / self.BASE_NORM

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
            v = (1 - norm**2)  # Smooth slow-down

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

    def angle_error_deg(self, pred, tgt):
        dot = (pred * tgt).sum(-1).clamp(-1, 1)
        return torch.acos(dot).detach().cpu().numpy() * 180 / math.pi

    def eval(self):
        self.motor.eval()
        self.gustatory.eval()
        CLS = ['Neutral', 'Appetitive', 'Aversive']

        ang_list, verr_list, ovl_list, spd_list, lbl_list = [], [], [], [], []

        with torch.no_grad():
            for odors, labels in tqdm(self.dataloader, desc="Evaluating Motor Module (Gustatory-driven)"):
                odors, labels = odors.to(self.device), labels.to(self.device)

                # Forward pass through frozen gustatory module
                feats = self.gustatory(odors)
                feats_soft = torch.softmax(feats, dim=-1)
                ovl = odors.norm(dim=1, keepdim=True) / self.BASE_NORM

                # Random current heading
                B = odors.size(0)
                angles = torch.rand(B, device=self.device) * 2 * math.pi
                h0 = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

                # Prepare input to motor module
                inp = torch.cat([feats, feats_soft, h0], dim=1)
                d_pred, v_pred = self.motor(inp)

                # Generate targets
                tgt = torch.stack([
                    self.sample_target(o, l.item(), a)
                    for o, l, a in zip(odors, labels, angles)
                ]).to(self.device)

                d_tgt, v_tgt = tgt[:, :2], tgt[:, 2]

                # Store metrics
                ang_error = self.angle_error_deg(d_pred, d_tgt)
                verr = (v_pred - v_tgt).cpu().numpy()
                spd = v_pred.cpu().numpy()
                ovl = ovl.cpu().numpy().squeeze()
                lbl = labels.cpu().numpy()

                ang_list.append(ang_error)
                verr_list.append(verr)
                spd_list.append(spd)
                ovl_list.append(ovl)
                lbl_list.append(lbl)

        # Combine all
        ang = np.concatenate(ang_list)
        verr = np.concatenate(verr_list)
        spd = np.concatenate(spd_list)
        ovl = np.concatenate(ovl_list)
        lbls = np.concatenate(lbl_list)

        # Print overall stats
        print(f"\n|Δθ| mean: {ang.mean():.2f}° ± {ang.std():.2f}")
        print(f"Speed MSE: {np.mean(verr ** 2):.4f}")
        r, p = pearsonr(ovl, spd)
        print(f"Overlap→Speed correlation: r = {r:.3f}, p = {p:.1e}")

        # Per-class breakdown
        for k, name in enumerate(CLS):
            sel = lbls == k
            print(f"{name}: |Δθ| {ang[sel].mean():.2f}°, Speed MSE {np.mean(verr[sel] ** 2):.4f}")