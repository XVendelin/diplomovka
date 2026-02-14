"""
SAC Robot Navigation Training
Main script for training a robot navigation agent using Soft Actor-Critic.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from image_extraction import druhy
from robot_env import RobotNavigationEnv
from robot_networks import ActorNetwork, CriticNetwork


class ReplayBuffer:
    """Experience replay buffer for SAC."""

    def __init__(self, capacity, obs_dim, action_dim, sequence_length=10):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample random batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    """Soft Actor-Critic agent."""

    def __init__(self, obs_dim, action_dim, device='cpu'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        # Networks
        self.actor = ActorNetwork(obs_dim).to(device)
        self.critic1 = CriticNetwork(obs_dim).to(device)
        self.critic2 = CriticNetwork(obs_dim).to(device)
        self.critic1_target = CriticNetwork(obs_dim).to(device)
        self.critic2_target = CriticNetwork(obs_dim).to(device)

        # Copy parameters to target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        # SAC parameters
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2  # Temperature parameter

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=int(1e6),
            obs_dim=obs_dim,
            action_dim=action_dim
        )

        self.batch_size = 2048
        self.warmup_steps = 5000
        self.update_count = 0

    def select_action(self, state, evaluate=False):
        """Select action from policy."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _ = self.actor.get_action(state, deterministic=evaluate)

        return action.cpu().numpy().squeeze()

    def update(self):
        """Update networks using SAC algorithm."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # === Update Critic ===
        with torch.no_grad():
            next_actions, next_log_probs, _ = self._get_action_and_log_prob(next_states)

            q1_next, _ = self.critic1_target(next_states, next_actions)
            q2_next, _ = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs

            q_target = rewards + (1 - dones) * self.gamma * q_next

        q1_pred, _ = self.critic1(states, actions)
        q2_pred, _ = self.critic2(states, actions)

        critic1_loss = nn.MSELoss()(q1_pred, q_target)
        critic2_loss = nn.MSELoss()(q2_pred, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()

        # === Update Actor ===
        new_actions, log_probs, _ = self._get_action_and_log_prob(states)

        q1_new, _ = self.critic1(states, new_actions)
        q2_new, _ = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # === Update target networks ===
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)

        self.update_count += 1

        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item()
        }

    def _get_action_and_log_prob(self, states):
        """Get action and log probability from policy."""
        mean, std, hidden = self.actor(states)

        dist = torch.distributions.Normal(mean, std)
        actions = dist.rsample()  # Reparameterization trick
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)

        # Clip actions
        actions = torch.clamp(actions, -10, 10)

        return actions, log_probs, hidden

    def _soft_update(self, target, source):
        """Soft update of target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def save(self, filepath):
        """Save agent."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
        }, filepath)

    def load(self, filepath):
        """Load agent."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])


def train_agent(env, agent, max_episodes=5000, max_steps_per_episode=500):
    """
    Train SAC agent.

    Parameters:
    -----------
    env : RobotNavigationEnv
        Environment
    agent : SACAgent
        SAC agent
    max_episodes : int
        Maximum number of episodes
    max_steps_per_episode : int
        Maximum steps per episode

    Returns:
    --------
    episode_rewards : list
        Reward per episode
    """
    episode_rewards = []
    episode_lengths = []
    avg_rewards = deque(maxlen=10)

    total_steps = 0

    print("\n=== Starting SAC Training ===")

    for episode in tqdm(range(max_episodes), desc="Training"):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(max_steps_per_episode):
            # Select action
            if total_steps < agent.warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Update agent
            if total_steps >= agent.warmup_steps:
                losses = agent.update()

            episode_reward += reward
            episode_length += 1
            total_steps += 1

            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        avg_rewards.append(episode_reward)
        """
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(avg_rewards)
            print(f"\nEpisode {episode + 1}/{max_episodes}")
            print(f"  Avg Reward (last 10): {avg_reward:.2f}")
            print(f"  Episode Length: {episode_length}")
            print(f"  Total Steps: {total_steps}")

        # Save checkpoint
        if (episode + 1) % 100 == 0:
            agent.save(f'checkpoints/agent_episode_{episode + 1}.pth')
        """

    return episode_rewards, episode_lengths


def main():
    """Main training function."""
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)

    # Load maps
    print("Loading maps...")
    coords1 = np.array([[280, 400], [280, 520], [400, 520], [400, 400]])
    map1 = druhy('image.jpg', coords1)

    coords2 = np.array([[475, 1100], [475, 1200], [575, 1200], [575, 1100]])
    map2 = druhy('image.jpg', coords2)

    res = 0.1  # map resolution [m/cell]
    obs_size = 20
    obs_radius = obs_size * res / 2
    dt = 0.05

    # Define routes
    routes1 = [
        {'start': np.array([10, 18]) * res, 'goal': np.array([100, 10]) * res},
        {'start': np.array([10, 38]) * res, 'goal': np.array([100, 28]) * res}
    ]

    routes2 = [
        {'start': np.array([85, 10]) * res, 'goal': np.array([69, 95]) * res},
        {'start': np.array([56, 6]) * res, 'goal': np.array([39, 92]) * res}
    ]

    # Create environment
    env_data = {
        'scenarios': [
            {'map': map1, 'routes': routes1},
            {'map': map2, 'routes': routes2}
        ],
        'res': res,
        'obs_radius': obs_radius,
        'obs_size': obs_size,
        'dt': dt
    }

    env = RobotNavigationEnv(env_data)

    # Create agent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(obs_dim, action_dim, device=device)

    # Train
    episode_rewards, episode_lengths = train_agent(
        env, agent,
        max_episodes=5000,
        max_steps_per_episode=500
    )

    # Save final model
    agent.save('checkpoints/agent_final.pth')

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Lengths')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    print("\nTraining complete!")


if __name__ == '__main__':
    main()