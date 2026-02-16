import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


def extract_local_terrain(x, y, map_array, res, radius, grid_size):
    half = grid_size // 2
    local_map = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            # World coordinates of this grid cell
            wx = x + (j - half) * radius / half
            wy = y + (i - half) * radius / half

            # Convert to map indices
            ix = int(round(wx / res))
            iy = int(round(wy / res))

            # Check bounds and assign value
            if 0 <= ix < map_array.shape[1] and 0 <= iy < map_array.shape[0]:
                local_map[j, i] = map_array[ix, iy]
            else:
                local_map[j, i] = 1.0  # Out of bounds = obstacle

    return local_map


def draw_tracked_robot(ax, x, y, theta, L, y_offset, res):
    # Robot corners in local frame
    corners_robot = np.array([
        [-L, -y_offset],
        [L, -y_offset],
        [L, y_offset],
        [-L, y_offset]
    ])

    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Transform to world frame
    corners_world = (R @ corners_robot.T).T
    corners_world[:, 0] += x
    corners_world[:, 1] += y

    # Convert to map coordinates (col, row)
    col = corners_world[:, 1] / res
    row = corners_world[:, 0] / res

    # Draw robot body
    polygon = MplPolygon(np.column_stack([col, row]),
                         facecolor='red',
                         edgecolor='red',
                         linewidth=0.1,
                         alpha=0.7)
    ax.add_patch(polygon)

    # Draw direction indicator (front)
    front_x = x + L * np.cos(theta)
    front_y = y + L * np.sin(theta)

    ax.plot([y / res, front_y / res],
            [x / res, front_x / res],
            'k-', linewidth=2)


def wrap_to_pi(angle):
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def calculate_reward(state, goal_pos, terrain, dist_to_goal, heading_error, res=0.1):
    sizex = int(np.ceil(0.5 / res / 2))
    sizey = int(np.ceil(0.5 / res / 2))
    center = terrain.shape[0] // 2

    # Extract robot hitbox
    hitbox = terrain[center - sizey:center + sizey + 1,
    center - sizex:center + sizex + 1]
    hit_count = np.sum(hitbox == 1)

    v = state[3]  # linear velocity

    # Distance penalty
    reward = -0.2 * dist_to_goal

    # Collision penalty
    reward -= 0 * hit_count

    # Heading alignment
    reward -= 0.1 * abs(heading_error)

    # Speed control based on terrain difficulty
    terrain_difficulty = np.mean(hitbox)
    safe_speed = 1 * (1 - terrain_difficulty)
    speed_penalty = 2 * (v - safe_speed) ** 2
    reward -= 0.5 * speed_penalty

    return reward