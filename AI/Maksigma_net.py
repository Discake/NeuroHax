import torch
import torch.nn as nn
import torch.nn.functional as F
import Constants

class Maksigma_net(nn.Module):
    def __init__(self, state_dim=Constants.state_size, hidden_dim=64):
        super().__init__()
        self.name = "AngleSpeed_Correct"
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # === ANGLE + SPEED HEADS ===
        self.angle_head = nn.Linear(hidden_dim, 2)    # [mean, log_std]
        self.speed_head = nn.Linear(hidden_dim, 2)    # [mean, log_std] 
        self.kick_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        x = self.shared(state)
        
        # Angle parameters
        angle_params = self.angle_head(x)  # [batch, 2]
        angle_mean = angle_params[:, 0]    # [batch]
        angle_log_std = angle_params[:, 1].clamp(-2, 1)  # [batch], ограничиваем
        
        # Speed parameters  
        speed_params = self.speed_head(x)  # [batch, 2]
        speed_mean = speed_params[:, 0]    # [batch]
        speed_log_std = speed_params[:, 1].clamp(-2, 1)  # [batch]
        
        kick_logit = self.kick_head(x).squeeze(-1)  # [batch]
        value = self.value_head(x).squeeze(-1)      # [batch]
        
        return angle_mean, angle_log_std, speed_mean, speed_log_std, kick_logit, value
    
    def get_action(self, state):
        angle_mean, angle_log_std, speed_mean, speed_log_std, kick_logit, value = self.forward(state)
        
        # === СОЗДАЕМ РАСПРЕДЕЛЕНИЯ ===
        angle_std = torch.exp(angle_log_std)
        angle_dist = torch.distributions.Normal(angle_mean, angle_std)
        
        # Для скорости используем Normal, а потом softplus для положительности
        speed_std = torch.exp(speed_log_std) 
        speed_dist = torch.distributions.Normal(speed_mean, speed_std)
        
        kick_dist = torch.distributions.Bernoulli(logits=kick_logit)
        
        # === СЕМПЛИРУЕМ ДЕЙСТВИЯ ===
        angle_raw = angle_dist.rsample()  # Сырой угол (может быть любым)
        speed_raw = speed_dist.rsample()  # Сырая скорость (может быть отрицательной)
        kick = kick_dist.sample()
        
        # === ПРИМЕНЯЕМ ПРЕОБРАЗОВАНИЯ ===
        # Нормализация угла в [-π, π]
        angle_normalized = torch.remainder(angle_raw + torch.pi, 2 * torch.pi) - torch.pi
        
        # Положительная скорость через softplus
        speed_positive = F.softplus(speed_raw).clamp(0.01, Constants.max_player_speed)
        
        # === КОНВЕРТИРУЕМ В ДЕКАРТОВЫ КООРДИНАТЫ ===
        velocity_x = speed_positive * torch.cos(angle_normalized)
        velocity_y = speed_positive * torch.sin(angle_normalized)
        
        # === СОХРАНЯЕМ СЫРЫЕ ЗНАЧЕНИЯ ДЛЯ LOG_PROB ===
        action_raw = torch.stack([angle_raw, speed_raw, kick], dim=-1)
        action_final = torch.stack([velocity_x, velocity_y, kick], dim=-1)
        
        # === LOG_PROB ОТ СЫРЫХ ЗНАЧЕНИЙ ===
        angle_logp = angle_dist.log_prob(angle_raw)
        speed_logp = speed_dist.log_prob(speed_raw)
        kick_logp = kick_dist.log_prob(kick)
        total_logp = angle_logp + speed_logp + kick_logp
        
        entropy = angle_dist.entropy() + speed_dist.entropy() + kick_dist.entropy()
        
        return action_final, action_raw, total_logp, value, entropy
    
    def evaluate_actions(self, states, actions_final, actions_raw):
        """
        КЛЮЧ: Принимаем ОБА - финальные действия И сырые параметры
        """
        angle_mean, angle_log_std, speed_mean, speed_log_std, kick_logit, values = self.forward(states)
        
        # Извлекаем сырые параметры (ТЕ ЖЕ, что использовались для log_prob)
        angle_raw = actions_raw[:, 0]
        speed_raw = actions_raw[:, 1] 
        kick_actions = actions_raw[:, 2]
        
        # === ВОССОЗДАЕМ ТЕ ЖЕ РАСПРЕДЕЛЕНИЯ ===
        angle_std = torch.exp(angle_log_std)
        angle_dist = torch.distributions.Normal(angle_mean, angle_std)
        
        speed_std = torch.exp(speed_log_std)
        speed_dist = torch.distributions.Normal(speed_mean, speed_std)
        
        kick_dist = torch.distributions.Bernoulli(logits=kick_logit)
        
        # === LOG_PROB ОТ ТЕХ ЖЕ СЫРЫХ ЗНАЧЕНИЙ ===
        angle_logp = angle_dist.log_prob(angle_raw)
        speed_logp = speed_dist.log_prob(speed_raw)
        kick_logp = kick_dist.log_prob(kick_actions)
        total_logp = angle_logp + speed_logp + kick_logp
        
        entropy = angle_dist.entropy() + speed_dist.entropy() + kick_dist.entropy()
        
        return total_logp, values, entropy
    
    def select_action(self, state):
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False
            
            action_final, action_raw, log_prob, _, _ = self.get_action(state)
            
            if squeeze_output:
                action_final = action_final.squeeze(0)
                action_raw = action_raw.squeeze(0)
                log_prob = log_prob.squeeze(0)
        
        return action_final, action_raw, log_prob  # Возвращаем ОБА!
