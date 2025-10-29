import torch
import torch.nn as nn

import Constants

class Maksigma_net(nn.Module):
    def __init__(self, state_dim=8, hidden_dim=64):
        self.name = "Maksigma_net_ravnykh"
        super().__init__()
        # Shared layers (feature extractor)
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # --- Actor Head for velocity (continuous 2D) ---
        self.actor_velocity = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.velocity_mean = nn.Linear(hidden_dim, 2)      # output: [mu_x, mu_y]
        self.velocity_log_sigma = nn.Linear(hidden_dim, 2) # output: [log_sigma_x, log_sigma_y]

        # --- Actor Head for kick (discrete) ---
        self.actor_kick = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.kick_logit = nn.Linear(hidden_dim, 1)         # output: логит (до sigmoid)

        # --- Critic Head (state value) ---
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),                      # output: скаляр value
        )

    def forward(self, state):
        x = self.shared(state)  # [batch, hidden]
        # Actor velocity
        av = self.actor_velocity(x)           # [batch, hidden]
        velocity_mean = self.velocity_mean(av)         # [batch, 2]
        velocity_log_sigma = self.velocity_log_sigma(av) # [batch, 2] (можно clamp для стабильности)

        # Actor kick (можно не выделять kick отдельный путь — для простых задач достаточно shared x)
        ak = self.actor_kick(x)
        kick_logit = self.kick_logit(ak).squeeze(-1)     # [batch] или [batch, 1]

        # Critic
        value = self.critic(x).squeeze(-1)               # [batch]

        return velocity_mean, velocity_log_sigma, kick_logit, value
    
    def get_action(self, state):
        # 1. Прямой проход через actor-critic сеть
        velocity_mean, log_sigma, kick_logit, value = self.forward(state)
        
        # 2. Создаём распределения
        sigma = torch.exp(log_sigma).clamp(min=0.01, max=1.5)
        velocity_dist = torch.distributions.Normal(velocity_mean, sigma)
        kick_dist = torch.distributions.Bernoulli(logits=kick_logit)
        
        # 3. СЕМПЛИРУЕМ действие (с шумом для exploration)
        velocity_raw = velocity_dist.rsample()
        kick = kick_dist.sample()
        
        # 4. Вычисляем log_prob ДО преобразований
        velocity_logp = velocity_dist.log_prob(velocity_raw).sum(-1)
        kick_logp = kick_dist.log_prob(kick)
        total_logp = velocity_logp + kick_logp
        total_entropy = velocity_dist.entropy().sum(-1) + kick_dist.entropy()
        
        # 5. Применяем ограничения (tanh/sigmoid)
        velocity_action = torch.tanh(velocity_raw)
        
        # 6. Формируем полное действие
        action = torch.cat([velocity_action, kick.unsqueeze(-1)], dim=-1)
        
        return action, total_logp, value, total_entropy
    
    def evaluate_actions(self, states, actions):
        # 1. Прямой проход через сеть (С ГРАДИЕНТАМИ!)
        velocity_mean, log_sigma, kick_logit, values = self.forward(states)
        
        # 2. Извлекаем компоненты действий
        velocity_actions = actions[:, :2]  # Первые 2 компоненты
        kick_actions = actions[:, 2]       # Третья компонента
        
        # 3. Создаём распределения с текущими параметрами
        sigma = torch.exp(log_sigma).clamp(min=0.01, max=1.5)
        velocity_dist = torch.distributions.Normal(velocity_mean, sigma)
        kick_dist = torch.distributions.Bernoulli(logits=kick_logit)
        
        # 4. ВАЖНО: НЕ семплируем, а вычисляем log_prob для СУЩЕСТВУЮЩИХ действий
        # Если действие прошло через tanh, нужно обратить преобразование
        velocity_raw = torch.atanh(velocity_actions.clamp(-0.9999, 0.9999))
        
        velocity_logp = velocity_dist.log_prob(velocity_raw).sum(-1)
        kick_logp = kick_dist.log_prob(kick_actions)
        total_logp = velocity_logp + kick_logp
        
        # 5. Вычисляем энтропию
        velocity_entropy = velocity_dist.entropy().sum(-1)
        kick_entropy = kick_dist.entropy()
        total_entropy = velocity_entropy + kick_entropy
        
        return total_logp, values.squeeze(-1), total_entropy