import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
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