import math

def update_position(position, heading, pred_direction, pred_velocity, 
                    max_turn_deg=30, max_step=15, arena_size=450):
    """
    Simulates one step of movement based on current heading and predicted motor output.

    Args:
        position (tuple): Current (x, y) position of the agent in the arena.
        heading (float): Current heading angle in radians.
        pred_direction (Tensor): 2D unit vector [cosθ, sinθ] from MotorModule.
        pred_velocity (float): Scalar velocity from MotorModule in [0, 1].
        max_turn_deg (float): Maximum turning angle per step in degrees.
        max_step (float): Maximum step length per frame at velocity = 1.
        arena_size (int): Size of the square arena.

    Returns:
        new_position (tuple): Updated (x, y) position.
        new_heading (float): Updated heading angle in radians.
    """

    # Convert predicted direction into target angle
    target_angle = math.atan2(pred_direction[1], pred_direction[0])
    
    # Calculate angular difference and clamp it to ±max_turn_deg
    dtheta = ((target_angle - heading + math.pi) % (2 * math.pi)) - math.pi
    max_turn_rad = math.radians(max_turn_deg)
    dtheta = max(-max_turn_rad, min(max_turn_rad, dtheta))  # Clamp to max turn

    # Update heading
    new_heading = (heading + dtheta) % (2 * math.pi)

    # Determine step size based on velocity
    step_size = pred_velocity * max_step

    # Update position
    dx = step_size * math.cos(new_heading)
    dy = step_size * math.sin(new_heading)
    new_x = min(max(position[0] + dx, 0), arena_size - 1)
    new_y = min(max(position[1] + dy, 0), arena_size - 1)

    return (new_x, new_y), new_heading