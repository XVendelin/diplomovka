"""
Neural network architectures for SAC agent.
Implements Actor and Critic networks with LSTM layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """
    Actor network for SAC with LSTM.
    Outputs mean and std for Gaussian policy.
    """
    def __init__(self, num_obs, hidden_size=256, lstm_size=128, num_actions=4):
        super(ActorNetwork, self).__init__()

        self.num_obs = num_obs
        self.lstm_size = lstm_size

        # Common pathway
        self.fc_pre_lstm = nn.Linear(num_obs, hidden_size)
        self.lstm = nn.LSTM(hidden_size, lstm_size, batch_first=True)
        self.fc_common = nn.Linear(lstm_size, 128)

        # Mean pathway
        self.fc_mean = nn.Linear(128, num_actions)

        # Std pathway
        self.fc_std = nn.Linear(128, num_actions)

    def forward(self, obs, hidden=None):
        """
        Forward pass.

        Parameters:
        -----------
        obs : torch.Tensor, shape (batch, seq_len, num_obs) or (batch, num_obs)
            Observations
        hidden : tuple of torch.Tensor or None
            LSTM hidden state

        Returns:
        --------
        mean : torch.Tensor
            Action mean
        std : torch.Tensor
            Action standard deviation
        hidden : tuple
            Updated LSTM hidden state
        """
        # Handle both (batch, num_obs) and (batch, seq_len, num_obs)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # Add sequence dimension

        # Common pathway
        x = F.relu(self.fc_pre_lstm(obs))
        x, hidden = self.lstm(x, hidden)
        x = F.relu(self.fc_common(x))

        # Mean pathway (scaled tanh)
        mean = torch.tanh(self.fc_mean(x)) * 10.0

        # Std pathway (softplus ensures positive)
        std = F.softplus(self.fc_std(x))

        # Squeeze sequence dimension for batch training
        mean = mean.squeeze(1)
        std = std.squeeze(1)

        return mean, std, hidden

    def get_action(self, obs, hidden=None, deterministic=False):
        """
        Sample action from policy.

        Parameters:
        -----------
        obs : torch.Tensor
            Observation
        hidden : tuple or None
            LSTM hidden state
        deterministic : bool
            If True, return mean action

        Returns:
        --------
        action : torch.Tensor
            Sampled action
        hidden : tuple
            Updated hidden state
        """
        mean, std, hidden = self.forward(obs, hidden)

        if deterministic:
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()

        # Clip to valid range
        action = torch.clamp(action, -10, 10)

        return action, hidden


class CriticNetwork(nn.Module):
    """
    Critic network for SAC with LSTM.
    Outputs Q-value given state and action.
    """
    def __init__(self, num_obs, num_actions=4, hidden_size=256, lstm_size=128):
        super(CriticNetwork, self).__init__()

        self.num_obs = num_obs
        self.lstm_size = lstm_size

        # State pathway
        self.fc_state = nn.Linear(num_obs, hidden_size)

        # Action pathway
        self.fc_action = nn.Linear(num_actions, hidden_size)

        # Common pathway
        self.lstm = nn.LSTM(hidden_size, lstm_size, batch_first=True)
        self.fc_common = nn.Linear(lstm_size, 128)
        self.fc_output = nn.Linear(128, 1)

    def forward(self, obs, action, hidden=None):
        """
        Forward pass.

        Parameters:
        -----------
        obs : torch.Tensor
            Observations
        action : torch.Tensor
            Actions
        hidden : tuple or None
            LSTM hidden state

        Returns:
        --------
        q_value : torch.Tensor
            Q-value estimate
        hidden : tuple
            Updated hidden state
        """
        # Handle both (batch, num_obs) and (batch, seq_len, num_obs)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        if action.dim() == 2:
            action = action.unsqueeze(1)

        # Separate pathways
        state_features = F.relu(self.fc_state(obs))
        action_features = F.relu(self.fc_action(action))

        # Combine
        x = state_features + action_features
        x, hidden = self.lstm(x, hidden)
        x = F.relu(self.fc_common(x))
        q_value = self.fc_output(x)

        # Squeeze sequence dimension for batch training
        q_value = q_value.squeeze(1)

        return q_value, hidden


def build_actor_network(num_obs):
    """
    Build actor network.

    Parameters:
    -----------
    num_obs : int
        Number of observation features

    Returns:
    --------
    network : ActorNetwork
        Actor network instance
    """
    return ActorNetwork(num_obs)


def build_critic_network(num_obs):
    """
    Build critic network.

    Parameters:
    -----------
    num_obs : int
        Number of observation features

    Returns:
    --------
    network : CriticNetwork
        Critic network instance
    """
    return CriticNetwork(num_obs)