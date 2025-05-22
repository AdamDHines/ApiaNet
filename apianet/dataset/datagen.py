'''
This script consists of classes and functions to generate synthetic data for training the ApiaNet system.

classes:
    - GustatoryDataset: Generates neutral, attractive, and aversive gustatory signals to train the gustatory and motor systems of ApiaNet.
'''


# Imports
import cv2
import torch
import random

import numpy as np

from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFilter, ImageChops

class VisionDataset(Dataset):
    """
    PyTorch dataset simulating a synthetic visual environment with distinct geometric shapes.
    Returns:
        - image (Tensor): RGB image patch of size 75x75
        - label (int): index of the dominant RGB channel (0=Red, 1=Green, 2=Blue)
        - edge_mask (Tensor): binary edge mask of the image patch

    Shapes are drawn with randomized spatial arrangements, and patches are extracted 
    to simulate a visual field. Labels are based on the dominant color used for all shapes.
    """
    def __init__(self, num_samples, image_transform=None, mask_transform=None, eval=False):
        self.num_samples = num_samples                # Number of samples in the dataset
        self.image_transform = image_transform        # Optional transform for image
        self.mask_transform = mask_transform          # Optional transform for mask

        # Arena configuration and shape placement parameters
        self.img_size = 450                           # Full image size
        self.patch_size = 75                          # Patch size to simulate visual field
        self.center = (self.img_size // 2, self.img_size // 2)  # Center of arena
        self.outer_radius = 200                       # Radius of the outer arena circle
        self.small_radius = 37.5                      # Radius of the inner reward zone
        self.buffer = 5                               # Distance buffer between reward and shapes
        self.border_width = 2                         # Outline thickness for reward zones
        self.min_distance = 10                        # Minimum spacing between shapes
        self.num_shapes = 400                         # Number of shapes to generate per scene
        self.shape_types = ['circle', 'square', 'triangle', 'cross']  # Supported shape types
        self.dot_size = 80                      # Size of the dots in eval mode
        self.num_dots = 400                         # Number of dots to generate in eval mode

        # Compute shape radius to occupy a consistent area
        self.shape_radius = self.compute_shape_radius(80)
        self.dot_radius = self.compute_shape_radius(self.dot_size)

        # green dot percentages
        self.green_percantage_options = [80, 20]  # Options for green dot percentage in eval mode

        # store eval flag
        self.eval = eval

    def __len__(self):
        return self.num_samples

    def compute_shape_radius(self, shape_size):
        # Compute radius from pixel area assuming circular equivalence
        return np.sqrt(shape_size / np.pi)
    
    def generate_random_circle_dots(self, num_dots, green_percentage, min_distance, exclude_radius, max_distance):
        xs, ys = [], []
        attempts = 0
        max_attempts = num_dots * 100
        while len(xs) < num_dots and attempts < max_attempts:
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(exclude_radius, max_distance)
            new_x = distance * np.cos(angle)
            new_y = distance * np.sin(angle)
            if all(np.hypot(new_x - xi, new_y - yi) >= min_distance for xi, yi in zip(xs, ys)):
                xs.append(new_x)
                ys.append(new_y)
            attempts += 1
        num_green = int(num_dots * green_percentage / 100)
        num_blue = num_dots - num_green
        colors = ['green'] * num_green + ['blue'] * num_blue
        random.shuffle(colors)
        return np.array(xs), np.array(ys), colors

    def generate_random_positions(self, num_items, min_distance, exclude_radius, max_distance):
        """
        Generates non-overlapping (x, y) offsets from center for placing shapes.
        Ensures shapes lie outside the inner reward area but within the outer arena.
        """
        xs, ys = [], []
        attempts = 0
        max_attempts = num_items * 100
        while len(xs) < num_items and attempts < max_attempts:
            angle = np.random.uniform(0, 2*np.pi)
            distance = np.random.uniform(exclude_radius, max_distance)
            new_x = distance * np.cos(angle)
            new_y = distance * np.sin(angle)
            if all(np.hypot(new_x - xi, new_y - yi) >= min_distance for xi, yi in zip(xs, ys)):
                xs.append(new_x)
                ys.append(new_y)
            attempts += 1
        return np.array(xs), np.array(ys)

    # Shape drawing functions for different types
    def draw_circle(self, draw_obj, cx, cy, radius, fill_color):
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        draw_obj.ellipse(bbox, fill=fill_color)

    def draw_square(self, draw_obj, cx, cy, radius, fill_color):
        draw_obj.polygon([
            (cx - radius, cy - radius),
            (cx + radius, cy - radius),
            (cx + radius, cy + radius),
            (cx - radius, cy + radius)
        ], fill=fill_color)

    def draw_triangle(self, draw_obj, cx, cy, radius, fill_color):
        h = radius * np.sqrt(3)
        draw_obj.polygon([
            (cx, cy - 2*h/3),
            (cx - radius, cy + h/3),
            (cx + radius, cy + h/3)
        ], fill=fill_color)

    def draw_cross(self, draw_obj, cx, cy, radius, fill_color):
        thickness = max(1, int(radius * 0.3))
        draw_obj.line([(cx - radius, cy - radius), (cx + radius, cy + radius)],
                      fill=fill_color, width=thickness)
        draw_obj.line([(cx - radius, cy + radius), (cx + radius, cy - radius)],
                      fill=fill_color, width=thickness)

    def draw_shape_at_position(self, draw_obj, shape_type, cx, cy, shape_radius, fill_color):
        # Dispatch to appropriate drawing method
        getattr(self, f"draw_{shape_type}")(draw_obj, cx, cy, shape_radius, fill_color)

    def generate_background_image(self):
        """
        Creates an off-white background with pink dots, skipping any that fall inside the arena.
        """
        bg = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 255
        dot_radius = 3
        for _ in range(4000):
            px, py = random.randint(0, self.img_size-1), random.randint(0, self.img_size-1)
            if (px - self.center[0])**2 + (py - self.center[1])**2 < self.outer_radius**2:
                continue  # Dot falls inside arena — skip it
            cv2.circle(bg, (px, py), dot_radius, (230, 161, 161), thickness=-1)
        return Image.fromarray(bg)

    def generate_full_stimulus(self):
        """
        Generates the full arena stimulus. If `self.eval` is True, use green/blue dot-based generation
        with label based on green percentage. Otherwise, use random shapes and label by RGB dominance.
        Both methods return:
            - The final RGB image
            - A label
            - A binary edge mask
        """
        # Start with background
        background_img = self.generate_background_image().convert("RGBA")
        img = Image.new("RGBA", (self.img_size, self.img_size))
        img.paste(background_img, (0, 0))
        draw = ImageDraw.Draw(img)

        # Draw arena structure
        draw.ellipse(
            [self.center[0]-self.outer_radius, self.center[1]-self.outer_radius,
            self.center[0]+self.outer_radius, self.center[1]+self.outer_radius],
            fill=(255,255,255,255), outline="black"
        )
        draw.ellipse(
            [self.center[0]-self.small_radius, self.center[1]-self.small_radius,
            self.center[0]+self.small_radius, self.center[1]+self.small_radius],
            fill=(255,255,255,255), outline="black"
        )

        if self.eval:
            # Eval mode: dot-based with green/blue split
            green_percentage = random.choice(self.green_percantage_options)
            label = 1 if green_percentage > 50 else 2

            exclude_radius = self.small_radius + self.buffer + self.dot_radius + self.border_width
            max_distance = self.outer_radius - self.dot_radius - self.border_width

            xs, ys, colors = self.generate_random_circle_dots(
                self.num_dots, green_percentage, self.min_distance,
                exclude_radius, max_distance
            )

            for x_offset, y_offset, col in zip(xs, ys, colors):
                dot_color = (0, 255, 0, 255) if col == 'green' else (0, 0, 255, 255)
                bbox = [
                    self.center[0] + x_offset - self.dot_radius,
                    self.center[1] + y_offset - self.dot_radius,
                    self.center[0] + x_offset + self.dot_radius,
                    self.center[1] + y_offset + self.dot_radius
                ]
                draw.ellipse(bbox, fill=dot_color)

        else:
            # Training mode: shape-based with RGB-dominant label
            R, G, B = [random.randint(0, 255) for _ in range(3)]
            shape_color = (R, G, B, 255)
            label = int(np.argmax([R, G, B]))

            exclude_radius = self.small_radius + self.buffer + self.shape_radius + self.border_width
            max_distance = self.outer_radius - self.shape_radius - self.border_width

            xs, ys = self.generate_random_positions(self.num_shapes, self.min_distance, exclude_radius, max_distance)

            self.shapes_drawn = []
            for x_offset, y_offset in zip(xs, ys):
                cx, cy = self.center[0] + x_offset, self.center[1] + y_offset
                shape_type = random.choice(self.shape_types)
                self.shapes_drawn.append((cx, cy, shape_type))
                self.draw_shape_at_position(draw, shape_type, cx, cy, self.shape_radius, shape_color)

        # Finalize image
        img = img.convert("RGB")

        # Create mask
        mask = Image.new("L", (self.img_size, self.img_size), 0)
        draw_mask = ImageDraw.Draw(mask)

        # Always include inner reward circle and outer ring edge
        draw_mask.ellipse(
            [self.center[0]-self.small_radius, self.center[1]-self.small_radius,
            self.center[0]+self.small_radius, self.center[1]+self.small_radius],
            fill=255
        )
        draw_mask.ellipse(
            [self.center[0]-self.outer_radius, self.center[1]-self.outer_radius,
            self.center[0]+self.outer_radius, self.center[1]+self.outer_radius],
            outline=255, width=2
        )

        if not self.eval:
            # Add drawn shapes to mask
            for cx, cy, shape_type in self.shapes_drawn:
                self.draw_shape_at_position(draw_mask, shape_type, cx, cy, self.shape_radius, 255)

        # Add background dot edges to mask
        bg_mask = self.generate_background_image().convert("L")
        bg_mask = bg_mask.point(lambda p: 255 if p < 250 else 0)
        mask = ImageChops.lighter(mask, bg_mask)

        return img, label, mask

    def __getitem__(self, idx):
        """
        Returns a random cropped 75x75 patch from a full stimulus image, along with:
            - the RGB patch,
            - the class label (dominant color),
            - and a binary edge mask from the segmentation.
        """
        full_img, label, full_mask = self.generate_full_stimulus()

        # Randomly crop a patch from the full arena
        left = random.randint(0, self.img_size - self.patch_size)
        upper = random.randint(0, self.img_size - self.patch_size)
        right, lower = left + self.patch_size, upper + self.patch_size
        patch_img = full_img.crop((left, upper, right, lower))
        patch_mask_pil = full_mask.crop((left, upper, right, lower))

        # Apply edge detection to the mask
        patch_edge_pil = patch_mask_pil.filter(ImageFilter.FIND_EDGES)

        # Apply optional transforms
        if self.image_transform:
            patch_img = self.image_transform(patch_img)
        if self.mask_transform:
            patch_edge = self.mask_transform(patch_edge_pil)
            patch_edge = (patch_edge > 0.5).float()

        return patch_img, label, patch_edge

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
    def __init__(self, num_samples, centre_prob=0.9):
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