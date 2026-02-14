"""
Simple test script without real-time visualization.
Creates a single plot at the end showing the full trajectory.
Much faster and no plot spamming issues.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from image_extraction import druhy
from robot_dynamics import tracked_robot_dynamics
from robot_utils import extract_local_terrain, draw_tracked_robot, wrap_to_pi
from robot_networks import ActorNetwork


def test_agent_simple(agent_path, map_array, test_start, test_goal, res,
                      obs_size, obs_radius, max_steps=2000):
    """
    Test trained agent without real-time visualization.
    Creates a single plot at the end.
    """
    # Load agent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    local_terrain_size = obs_size * obs_size
    num_obs = 7 + local_terrain_size

    actor = ActorNetwork(num_obs).to(device)
    checkpoint = torch.load(agent_path, map_location=device)
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()

    # Initialize state
    dt = 0.05
    state = np.array([test_start[0], test_start[1], 0, 0, 0, 0, 0], dtype=np.float32)
    trajectory = [state[:2].copy()]

    success = False

    print(f"Starting test: {test_start} -> {test_goal}")
    print(f"Distance: {np.linalg.norm(test_goal - test_start):.2f}m")

    for step in range(max_steps):
        x, y = state[0], state[1]
        theta = state[2]

        # Extract local terrain
        local_terrain = extract_local_terrain(x, y, map_array, res, obs_radius, obs_size)
        terrain_vec = local_terrain.flatten()

        # Calculate goal info
        goal_dx = test_goal[0] - x
        goal_dy = test_goal[1] - y
        goal_heading_error = wrap_to_pi(np.arctan2(goal_dy, goal_dx) - theta)
        dist_to_goal = np.hypot(goal_dx, goal_dy)

        # Assemble observation
        obs = np.concatenate([
            [np.sin(theta)],
            [np.cos(theta)],
            [np.sin(goal_heading_error)],
            [np.cos(goal_heading_error)],
            [dist_to_goal / np.linalg.norm(test_goal - test_start)],
            [state[3] / 5],
            [state[4] / 5],
            terrain_vec
        ]).astype(np.float32)

        # Get action from trained agent
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _ = actor.get_action(obs_tensor, deterministic=True)
            M = action.cpu().numpy().squeeze()

        M = np.clip(M, -10, 10)

        # Update dynamics
        state = tracked_robot_dynamics(state, M, dt)
        trajectory.append(state[:2].copy())

        # Print progress every 100 steps
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{max_steps} | Distance to goal: {dist_to_goal:.2f}m")

        # Check success
        if dist_to_goal < 0.3:
            print(f'\n✓ SUCCESS! Goal reached in {step} steps ({dist_to_goal:.2f}m)')
            success = True
            break

        # Check stuck
        if step > 200 and len(trajectory) > 50:
            recent_displacement = np.linalg.norm(
                trajectory[-1] - trajectory[-50]
            )
            if recent_displacement < 0.5:
                print(f'\n✗ STUCK at step {step} ({dist_to_goal:.2f}m from goal)')
                break

    if not success and step >= max_steps - 1:
        print(f'\n✗ TIMEOUT: Max steps reached. Distance: {dist_to_goal:.2f}m')

    return success, np.array(trajectory)


def plot_trajectory(map_array, trajectory, test_start, test_goal, res, success):
    """
    Create a single plot showing the trajectory.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Map with trajectory
    ax = axes[0]
    ax.imshow(map_array.T, cmap='gray', origin='lower')
    ax.plot(trajectory[:, 1] / res, trajectory[:, 0] / res,
           'g-', linewidth=2, label='Trajectory')
    ax.plot(test_start[1] / res, test_start[0] / res,
           'go', markersize=10, label='Start')
    ax.plot(test_goal[1] / res, test_goal[0] / res,
           'r*', markersize=15, label='Goal')

    # Draw robot at final position
    if len(trajectory) > 0:
        final_pos = trajectory[-1]
        theta = 0  # We don't store theta in simple version
        draw_tracked_robot(ax, final_pos[0], final_pos[1], theta, 0.25, 0.25, res)

    status = "SUCCESS" if success else "FAILED"
    ax.set_title(f'Trajectory ({status}) - {len(trajectory)} steps')
    ax.set_xlabel('Y [cells]')
    ax.set_ylabel('X [cells]')
    ax.legend()
    ax.axis('equal')

    # Plot 2: Distance to goal over time
    ax = axes[1]
    distances = [np.linalg.norm(traj_point - test_goal) for traj_point in trajectory]
    ax.plot(distances, 'b-', linewidth=2)
    ax.axhline(y=0.3, color='r', linestyle='--', label='Goal threshold (0.3m)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Distance to Goal [m]')
    ax.set_title('Distance to Goal Over Time')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('test_trajectory.png', dpi=150)
    print("\n✓ Plot saved as 'test_trajectory.png'")
    plt.show()


def main():
    """Main testing function."""
    print("\n" + "="*60)
    print("Testing Trained Agent (Simple Version)")
    print("="*60)

    # Map selection
    map_choice = 2  # 1 or 2

    res = 0.1
    obs_size = 20
    obs_radius = obs_size * res / 2

    if map_choice == 1:
        print("Loading Map 1...")
        coords = np.array([[280, 400], [280, 520], [400, 520], [400, 400]])
        test_map = druhy('image.jpg', coords)
        test_start = np.array([10, 18]) * res
        test_goal = np.array([100, 10]) * res
    else:
        print("Loading Map 2...")
        coords = np.array([[475, 1100], [475, 1200], [575, 1200], [575, 1100]])
        test_map = druhy('image.jpg', coords)
        test_start = np.array([85, 10]) * res
        test_goal = np.array([69, 95]) * res

    print(f"Map shape: {test_map.shape}")

    # Test agent
    success, trajectory = test_agent_simple(
        agent_path='checkpoints/agent_final.pth',
        map_array=test_map,
        test_start=test_start,
        test_goal=test_goal,
        res=res,
        obs_size=obs_size,
        obs_radius=obs_radius,
        max_steps=2000
    )

    # Plot results
    plot_trajectory(test_map, trajectory, test_start, test_goal, res, success)

    # Save trajectory
    np.save('test_trajectory.npy', trajectory)
    print("✓ Trajectory saved to 'test_trajectory.npy'")

    print("\n" + "="*60)
    print(f"Test Complete: {'SUCCESS' if success else 'FAILED'}")
    print(f"Trajectory length: {len(trajectory)} steps")
    print("="*60)


if __name__ == '__main__':
    main()