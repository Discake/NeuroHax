import torch
import torch.nn as nn
import torch.nn.functional as F

import Constants

class UltraSimplePolicy(nn.Module):
    """Политика с обучаемым стандартным отклонением"""
    def __init__(self, state_dim=Constants.state_size):
        super().__init__()
        
        hidden_dim = 128
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
        )
        
        # Выходные головы
        self.velocity_mean_head = nn.Linear(hidden_dim // 2, 2)      # velocity mean
        self.velocity_std_head = nn.Linear(hidden_dim // 2, 2)       # velocity log_std (ОБУЧАЕМАЯ!)
        self.kick_head = nn.Linear(hidden_dim // 2, 1)              # kick logit  
        self.value_head = nn.Linear(hidden_dim // 2, 1)             # value
        
        self._init_weights()
        
    def _init_weights(self):
        """Консервативная инициализация"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # === СПЕЦИАЛЬНАЯ ИНИЦИАЛИЗАЦИЯ ДЛЯ STD HEAD ===
        # Инициализируем std head так, чтобы начальная std была разумной
        nn.init.constant_(self.velocity_std_head.weight, 0.0)
        nn.init.constant_(self.velocity_std_head.bias, -1.0)  # exp(-1) = 0.37
    
    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
            
        x = self.shared(state)
        
        # === УПРОЩЁННЫЕ ВЫХОДЫ БЕЗ TANH ===
        velocity_mean = self.velocity_mean_head(x)
        velocity_log_std = self.velocity_std_head(x)

        if self.training:
            velocity_log_std = torch.clamp(velocity_log_std, -2, 1)
        else:
            velocity_log_std = torch.clamp(velocity_log_std, -2, -1.9)
        
        kick_logit = self.kick_head(x).squeeze(-1)
        value = self.value_head(x).squeeze(-1)
        
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
        
        # === ПРЯМОЙ СЭМПЛИНГ БЕЗ TANH ===
        velocity = velocity_dist.rsample()
        
        # Ограничиваем действия в разумных пределах
        velocity = torch.clamp(velocity, -3, 3)  # Или используйте ваш max_velocity
        
        kick = kick_dist.sample()
        
        # Формируем action
        if velocity.dim() == 1:
            action = torch.cat([velocity, kick.unsqueeze(-1)], dim=-1)
        else:
            action = torch.cat([velocity, kick.unsqueeze(-1)], dim=-1)
        
        # === ПРОСТЫЕ LOG PROBS ===
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
        
        # === ПРЯМОЕ ВЫЧИСЛЕНИЕ БЕЗ ATANH ===
        velocity_logp = velocity_dist.log_prob(velocity_actions).sum(-1)
        kick_logp = kick_dist.log_prob(kick_actions)
        total_logp = velocity_logp + kick_logp
        
        entropy = velocity_dist.entropy().sum(-1) + kick_dist.entropy()
        
        return total_logp, values, entropy
    
    def select_action(self, state):
        """For data collection"""
        with torch.no_grad():
            action, log_prob, _, _ = self.get_action(state)
        return action, log_prob
    
    def get_std_info(self, state):
        """Диагностическая функция для мониторинга std"""
        with torch.no_grad():
            _, velocity_log_std, _, _ = self.forward(state)
            velocity_std = torch.exp(velocity_log_std)
            return {
                'log_std': velocity_log_std.cpu().numpy(),
                'std': velocity_std.cpu().numpy(),
                'std_x': velocity_std[0].item() if velocity_std.dim() > 0 else velocity_std.item(),
                'std_y': velocity_std[1].item() if velocity_std.dim() > 1 else velocity_std.item()
            }
