import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import Constants

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import Constants


class SeparateNetworkPolicy(nn.Module):
    """УПРОЩЁННАЯ архитектура с акцентом на value diversity"""
    def __init__(self, state_dim=Constants.state_size):
        super().__init__()
        
        hidden_dim = 256
        
        # === POLICY NETWORK (без изменений) ===
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
        )
        
        self.velocity_mean_head = nn.Linear(hidden_dim // 2, 2)
        self.velocity_log_std_head = nn.Linear(hidden_dim // 2, 2)
        self.kick_head = nn.Linear(hidden_dim // 2, 1)
        
        # === VALUE NETWORK - ПРОСТАЯ И ШИРОКАЯ ===
        # Меньше слоёв, больше нейронов в каждом
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        
        # === ДВОЙНОЙ VALUE HEAD ДЛЯ РАЗНООБРАЗИЯ ===
        self.value_head_1 = nn.Linear(128, 1)
        self.value_head_2 = nn.Linear(128, 1)
        
        self._init_weights()
        
    def _init_weights(self):
        """Правильная инициализация для разных активаций"""

        # Policy network - Orthogonal для Tanh
        for m in self.policy_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

        # === VALUE NETWORK - He initialization для ReLU ===
        for m in self.value_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.01)

        # Policy heads
        nn.init.orthogonal_(self.velocity_mean_head.weight, gain=0.01)
        nn.init.orthogonal_(self.velocity_log_std_head.weight, gain=0.01)
        nn.init.orthogonal_(self.kick_head.weight, gain=0.01)

        nn.init.zeros_(self.velocity_mean_head.bias)
        nn.init.constant_(self.velocity_log_std_head.bias, -0.7)
        nn.init.zeros_(self.kick_head.bias)

        # === VALUE HEADS - РАЗНАЯ инициализация для diversity ===
        nn.init.xavier_uniform_(self.value_head_1.weight, gain=2.0)
        nn.init.xavier_uniform_(self.value_head_2.weight, gain=0.5)

        nn.init.constant_(self.value_head_1.bias, 5.0)  # Оптимистичный
        nn.init.constant_(self.value_head_2.bias, -5.0) # Пессимистичный

    
    def forward(self, state, debug=False):




        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        # Policy forward
        policy_features = self.policy_net(state)
        velocity_mean = self.velocity_mean_head(policy_features)
        velocity_log_std = self.velocity_log_std_head(policy_features)

        velocity_log_std = torch.clamp(velocity_log_std, -0.5, 1)
        
        kick_logit = self.kick_head(policy_features).squeeze(-1)
        
        # Value forward
        value_features = self.value_net(state)
        
        # === ENSEMBLE OF TWO HEADS ===
        value_1 = self.value_head_1(value_features)
        value_2 = self.value_head_2(value_features)
        
        # Среднее из двух
        value = (value_1 + value_2) / 2.0
        value = value.squeeze(-1)
        
        # === ДИАГНОСТИКА ===
        if debug and torch.rand(1) < 0.1:  # 10% времени
            print(f"\n🔍 Value Network Debug:")
            print(f"   Features: mean={value_features.mean():.3f}, std={value_features.std():.3f}, "
                  f"range=[{value_features.min():.2f}, {value_features.max():.2f}]")
            print(f"   Value_1: mean={value_1.mean():.3f}, std={value_1.std():.3f}")
            print(f"   Value_2: mean={value_2.mean():.3f}, std={value_2.std():.3f}")
            print(f"   Final value: mean={value.mean():.3f}, std={value.std():.3f}")
        
        if single_sample:
            velocity_mean = velocity_mean.squeeze(0)
            velocity_log_std = velocity_log_std.squeeze(0)
            kick_logit = kick_logit.squeeze(0) if kick_logit.dim() > 0 else kick_logit
            value = value.squeeze(0) if value.dim() > 0 else value
        
        return velocity_mean, velocity_log_std, kick_logit, value
    
    def get_action(self, state):
        velocity_mean, velocity_log_std, kick_logit, value = self.forward(state)
        
        velocity_std = torch.exp(velocity_log_std)
        
        velocity_dist = torch.distributions.Normal(velocity_mean, velocity_std)
        kick_dist = torch.distributions.Bernoulli(logits=kick_logit)
        
        velocity = velocity_dist.rsample()
        velocity = torch.clamp(velocity, -3, 3)
        
        kick = kick_dist.sample()
        
        if velocity.dim() == 1:
            action = torch.cat([velocity, kick.unsqueeze(-1)], dim=-1)
        else:
            action = torch.cat([velocity, kick.unsqueeze(-1)], dim=-1)
        
        velocity_logp = velocity_dist.log_prob(velocity).sum(-1)
        kick_logp = kick_dist.log_prob(kick)
        total_logp = velocity_logp + kick_logp
        
        entropy = velocity_dist.entropy().sum(-1) + kick_dist.entropy()
        
        return action, total_logp, value, entropy
    
    def evaluate_actions(self, states, actions):
        velocity_mean, velocity_log_std, kick_logit, values = self.forward(states)
        
        velocity_actions = actions[:, :2]
        kick_actions = actions[:, 2]
        
        velocity_std = torch.exp(velocity_log_std)
        velocity_dist = torch.distributions.Normal(velocity_mean, velocity_std)
        kick_dist = torch.distributions.Bernoulli(logits=kick_logit)
        
        velocity_logp = velocity_dist.log_prob(velocity_actions).sum(-1)
        kick_logp = kick_dist.log_prob(kick_actions)
        total_logp = velocity_logp + kick_logp
        
        entropy = velocity_dist.entropy().sum(-1) + kick_dist.entropy()
        
        return total_logp, values, entropy
        
    def get_value(self, state):
        """For data collection"""
        _, _, _, value = self.forward(state)
        return value
    
    def select_action(self, state):
        """For data collection"""
        action, log_prob, _, _ = self.get_action(state)
        return action, log_prob