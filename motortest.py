import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math, random
from tqdm import tqdm

from apianet.src.modules import GustatoryModule, MotorModule

def update_position(position, heading, pred_direction, pred_velocity,
                    max_turn_deg=90, max_step=10, arena_size=(900, 450)):
    target_angle = math.atan2(pred_direction[1], pred_direction[0])
    dtheta = ((target_angle - heading + math.pi) % (2 * math.pi)) - math.pi
    max_turn_rad = math.radians(max_turn_deg)
    dtheta = max(-max_turn_rad, min(max_turn_rad, dtheta))
    new_heading = (heading + dtheta) % (2 * math.pi)
    step_size = pred_velocity * max_step
    dx = step_size * math.cos(new_heading)
    dy = step_size * math.sin(new_heading)
    new_x = min(max(position[0] + dx, 0), arena_size[0] - 1)
    new_y = min(max(position[1] + dy, 0), arena_size[1] - 1)
    return (new_x, new_y), new_heading

# === SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gustatory = GustatoryModule().to(device)
motor = MotorModule().to(device)

#load the weights
gustatory.load_state_dict(torch.load("./apianet/models/GustatoryModel.pth"))
motor.load_state_dict(torch.load("./apianet/models/MotorModel.pth"))

arena_width, arena_height = 900, 450
stim_radius = 75
base_gust = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float32)

att_pos = np.array([525, arena_height // 2])
avv_pos = np.array([225, arena_height // 2])

def get_stimulus_label_and_vector(pos):
    pos = np.array(pos)
    if np.linalg.norm(pos - att_pos) < stim_radius:
        overlap = 1 - np.linalg.norm(pos - att_pos) / stim_radius
        return 1, torch.tensor(base_gust * overlap, dtype=torch.float32)
    elif np.linalg.norm(pos - avv_pos) < stim_radius:
        overlap = 1 - np.linalg.norm(pos - avv_pos) / stim_radius
        return 2, torch.tensor(-base_gust * overlap, dtype=torch.float32)
    else:
        return 0, torch.zeros(5, dtype=torch.float32)

# === SIMULATION ===
pos = (arena_width / 2, arena_height / 2)
heading = random.uniform(-2, 2 * math.pi)
trajectory = [pos]

for _ in tqdm(range(1000), desc="Simulating trajectory"):
    label, odor = get_stimulus_label_and_vector(pos)
    odor = odor.to(device).unsqueeze(0)

    with torch.no_grad():
        gust_output = gustatory(odor)
        gust_softmax = torch.softmax(gust_output, dim=1) if gust_output.ndim == 2 else gust_output
        heading_vector = torch.tensor([[math.cos(heading), math.sin(heading)]], device=device)
        motor_input = torch.cat([gust_output, gust_softmax, heading_vector], dim=1)
        pred_dir, pred_vel = motor(motor_input)

    pos, heading = update_position(pos, heading, pred_dir[0].cpu().numpy(), float(pred_vel[0]), arena_size=(arena_width, arena_height))
    trajectory.append(pos)


# === PLOTTING ===
x, y = zip(*trajectory)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x, y, lw=1.5)
ax.add_patch(Circle(att_pos, stim_radius, color='green', alpha=0.3, label='Attractive'))
ax.add_patch(Circle(avv_pos, stim_radius, color='red', alpha=0.3, label='Aversive'))
ax.set_xlim(0, arena_width)
ax.set_ylim(0, arena_height)
ax.set_aspect('equal')
ax.set_title("Agent Navigation Trajectory")
ax.legend()
plt.tight_layout()
plt.show()

import matplotlib.patches as patches
import matplotlib.animation as animation
fig, ax = plt.subplots(figsize=(12, 6))

# Static background
ax.add_patch(Circle(att_pos, stim_radius, color='green', alpha=0.3, label='Attractive'))
ax.add_patch(Circle(avv_pos, stim_radius, color='red', alpha=0.3, label='Aversive'))

agent_size = 75
rect = patches.Rectangle((0, 0), agent_size, agent_size, linewidth=1.5,
                         edgecolor='blue', facecolor='blue', alpha=0.6)
ax.add_patch(rect)

path_line, = ax.plot([], [], color='black', lw=1.5, alpha=0.7)
ax.set_xlim(0, arena_width)
ax.set_ylim(0, arena_height)
ax.set_aspect('equal')
ax.set_title("Agent Movement Animation")
ax.legend()

def init():
    rect.set_xy((trajectory[0][0] - agent_size / 2, trajectory[0][1] - agent_size / 2))
    path_line.set_data([], [])
    return rect, path_line

def update(frame):
    x, y = trajectory[frame]
    rect.set_xy((x - agent_size / 2, y - agent_size / 2))
    path_line.set_data(*zip(*trajectory[:frame+1]))
    return rect, path_line

ani = animation.FuncAnimation(
    fig, update, frames=len(trajectory), init_func=init,
    interval=30, blit=True, repeat=False
)

# Save locally if needed:
# ani.save("agent_trajectory.mp4", fps=30, dpi=150)

plt.show()