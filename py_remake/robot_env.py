"""
Custom RL Environment for robot navigation.
Implements Gym-like interface for training with SAC.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from robot_dynamics import tracked_robot_dynamics
from robot_utils import extract_local_terrain, calculate_reward, wrap_to_pi


class RobotNavigationEnv(gym.Env):
    """
    Robot navigation environment for reinforcement learning.
    """

    def __init__(self, env_data):
        """
        Initialize environment.

        Parameters:
        -----------
        env_data : dict
            Contains:
            - scenarios: list of dicts with 'map' and 'routes'
            - res: map resolution
            - obs_radius: observation radius
            - obs_size: observation grid size
            - dt: time step
        """
        super(RobotNavigationEnv, self).__init__()

        self.env_data = env_data
        self.scenarios = env_data['scenarios']
        self.res = env_data['res']
        self.obs_radius = env_data['obs_radius']
        self.obs_size = env_data['obs_size']
        self.dt = env_data['dt']

        # Define observation and action spaces
        local_terrain_size = self.obs_size * self.obs_size
        num_obs = 7 + local_terrain_size

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_obs,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-10,
            high=10,
            shape=(4,),
            dtype=np.float32
        )

        # Episode tracking
        self.state = None
        self.trajectory = None
        self.step_count = 0
        self.current_map = None
        self.start_pos = None
        self.goal_pos = None
        self.max_steps = 500

    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.

        Returns:
        --------
        observation : np.ndarray
            Initial observation
        info : dict
            Additional information
        """
        super().reset(seed=seed)

        # Pick random scenario
        scenario_id = np.random.randint(len(self.scenarios))
        scenario = self.scenarios[scenario_id]

        # Pick random route
        route_id = np.random.randint(len(scenario['routes']))
        route = scenario['routes'][route_id]

        # Optional flip (swap start/goal)
        if np.random.rand() < 0.5:
            self.start_pos = route['start'].copy()
            self.goal_pos = route['goal'].copy()
            flipped = False
        else:
            self.start_pos = route['goal'].copy()
            self.goal_pos = route['start'].copy()
            flipped = True

        self.current_map = scenario['map']

        # Initialize robot with random orientation
        init_theta = np.random.uniform(-np.pi, np.pi)
        self.state = np.array([
            self.start_pos[0],
            self.start_pos[1],
            init_theta,
            0, 0, 0, 0
        ], dtype=np.float32)

        self.trajectory = [self.start_pos.copy()]
        self.step_count = 0

        # Get initial observation
        observation = self._get_observation()

        info = {
            'scenario_id': scenario_id,
            'route_id': route_id,
            'flipped': flipped
        }

        return observation, info

    def step(self, action):
        """
        Execute one step in the environment.

        Parameters:
        -----------
        action : np.ndarray
            Motor torques [M_FL, M_FR, M_RL, M_RR]

        Returns:
        --------
        observation : np.ndarray
            Next observation
        reward : float
            Reward signal
        terminated : bool
            Whether episode is done
        truncated : bool
            Whether episode was truncated
        info : dict
            Additional information
        """
        # Clip action to valid range
        action = np.clip(action, -10, 10)

        # Update dynamics
        self.state = tracked_robot_dynamics(self.state, action, self.dt)

        # Update trajectory
        self.trajectory.append(self.state[:2].copy())
        self.step_count += 1

        # Get observation
        observation = self._get_observation()

        # Calculate distances and errors
        x, y = self.state[0], self.state[1]
        theta = self.state[2]

        goal_dx = self.goal_pos[0] - x
        goal_dy = self.goal_pos[1] - y
        dist_to_goal = np.hypot(goal_dx, goal_dy)
        heading_error = wrap_to_pi(np.arctan2(goal_dy, goal_dx) - theta)

        # Extract local terrain
        local_terrain = extract_local_terrain(
            x, y, self.current_map,
            self.res, self.obs_radius, self.obs_size
        )

        # Calculate reward
        reward = calculate_reward(
            self.state, self.goal_pos, local_terrain,
            dist_to_goal, heading_error, self.res
        )

        # Check termination conditions
        terminated = False
        truncated = False

        # Success
        if dist_to_goal < 0.3:
            reward += 5000
            terminated = True

        # Max steps
        elif self.step_count >= self.max_steps:
            truncated = True

        # Invalid state
        elif np.any(np.isnan(self.state)) or np.any(np.isinf(self.state)):
            reward -= 100
            terminated = True

        # Stuck detection
        elif len(self.trajectory) > 100:
            recent_displacement = np.linalg.norm(
                np.array(self.trajectory[-1]) - np.array(self.trajectory[-50])
            )
            if recent_displacement < 0.3:
                reward -= 50
                terminated = True

        info = {
            'dist_to_goal': dist_to_goal,
            'step_count': self.step_count
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """
        Get current observation.

        Returns:
        --------
        obs : np.ndarray
            Observation vector
        """
        x = self.state[0]
        y = self.state[1]
        theta = self.state[2]
        v = self.state[3]
        omega = self.state[4]

        # Extract local terrain
        local_terrain = extract_local_terrain(
            x, y, self.current_map,
            self.res, self.obs_radius, self.obs_size
        )
        terrain_vec = local_terrain.flatten()

        # Goal information
        goal_dx = self.goal_pos[0] - x
        goal_dy = self.goal_pos[1] - y
        dist_to_goal = np.hypot(goal_dx, goal_dy)
        heading_error = wrap_to_pi(np.arctan2(goal_dy, goal_dx) - theta)

        # Normalize distance
        total_dist = np.linalg.norm(self.goal_pos - self.start_pos)
        normalized_dist = dist_to_goal / total_dist if total_dist > 0 else 0

        # Assemble observation
        obs = np.concatenate([
            [np.sin(theta)],
            [np.cos(theta)],
            [np.sin(heading_error)],
            [np.cos(heading_error)],
            [normalized_dist],
            [v / 5.0],
            [omega / 5.0],
            terrain_vec
        ]).astype(np.float32)

        return obs

    def render(self, mode='human'):
        """
        Render the environment (not implemented).
        """
        pass

    def close(self):
        """
        Clean up resources.
        """
        pass