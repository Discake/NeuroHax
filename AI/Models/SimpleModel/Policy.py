import torch
import torch.nn as nn

from AI.GameTranslation.TranslatorToPolicy import TranslatorToPolicy

class SimpleModel(nn.Module):
    def __init__(self, translator : TranslatorToPolicy):
        super().__init__()

        self.translator = translator

        state_dim = translator.get_state_dim()

        # Actor backbone (только для actor-голов)
        self.actor_backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Critic backbone (только для value head)
        self.critic_backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Actor heads
        self.velocity_head = nn.Linear(64, 4)
        self.kick_head = nn.Linear(64, 1)

        # Critic head
        self.value_head = nn.Linear(64, 1)


    def actor_parameters(self):
        """Параметры актора (actor_backbone + головы)"""
        return (list(self.actor_backbone.parameters()) +
                list(self.velocity_head.parameters()) +
                list(self.kick_head.parameters()))

    def critic_parameters(self):
        """Параметры критика (critic_backbone + value_head)"""
        return (list(self.critic_backbone.parameters()) +
                list(self.value_head.parameters()))

    def forward(self, state):
        actor_feat = self.actor_backbone(state)
        critic_feat = self.critic_backbone(state)

        velocity = self.velocity_head(actor_feat)
        kick_logit = self.kick_head(actor_feat)
        value = self.value_head(critic_feat)

        return velocity, kick_logit, value

    def actor_forward(self, state):
        """Только actor-путь — без critic backbone. Используется при сборе данных."""
        actor_feat = self.actor_backbone(state)
        return self.velocity_head(actor_feat), self.kick_head(actor_feat)

    def get_value(self, state):
        """Получение value для PPO (только critic path)"""
        if isinstance(state, dict):
            state = self.translator.translate(state)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        critic_feat = self.critic_backbone(state)
        return self.value_head(critic_feat).squeeze(-1)

    def get_action(self, state):
        # Если уже тензор (из env.step) — не перетранслируем, иначе берём из translator
        if not torch.is_tensor(state):
            state = self.translator.translate(state)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Только actor-путь: critic backbone не нужен при сборе данных
        move_logits, kick_logit = self.actor_forward(state)     # [B,4], [B,1]

        # Семплируем через sigmoid + bernoulli — без создания Distribution объектов
        move_probs = torch.sigmoid(move_logits)                 # [B,4]
        kick_prob  = torch.sigmoid(kick_logit)                  # [B,1]

        move = torch.bernoulli(move_probs)                      # [B,4]
        kick = torch.bernoulli(kick_prob)                       # [B,1]

        action = torch.cat([move, kick], dim=-1)                # [B,5]

        # log_prob без Distribution объекта
        move_logp = (move * torch.log(move_probs + 1e-8) +
                     (1 - move) * torch.log(1 - move_probs + 1e-8)).sum(-1)  # [B]
        kick_logp = (kick * torch.log(kick_prob + 1e-8) +
                     (1 - kick) * torch.log(1 - kick_prob + 1e-8)).squeeze(-1)  # [B]
        logp = move_logp + kick_logp

        if action.size(0) == 1:
            return action.squeeze(0), logp.squeeze(0)
        return action, logp
    
    def evaluate_actions(self, states, actions):
        """
        Вычисление log_prob и value для данных действий
        Требуется для PPO
        """
        if states.dim() == 1:
            states = states.unsqueeze(0)

        move_logits, kick_logit, values = self.forward(states)

        # Разделяем действия на movement и kick
        move_actions = actions[:, :4]   # [B, 4]
        kick_actions = actions[:, 4:5]  # [B, 1] — сохраняем размерность, иначе broadcasting даст [B, B]

        # Вычисляем распределения
        move_dist = torch.distributions.Bernoulli(logits=move_logits)
        kick_dist = torch.distributions.Bernoulli(logits=kick_logit)

        # Вычисляем log_prob
        move_logp = move_dist.log_prob(move_actions).sum(-1)          # [B]
        kick_logp = kick_dist.log_prob(kick_actions).squeeze(-1)      # [B]
        total_logp = move_logp + kick_logp                            # [B]

        # Вычисляем энтропию
        entropy = move_dist.entropy().sum(-1) + kick_dist.entropy().squeeze(-1)  # [B]

        return total_logp, values.squeeze(-1), entropy

    def select_action(self, state, deterministic=False):
        """
        Выбор действия (алиас для get_action, совместимость с визуализацией)
        
        Args:
            state: Состояние (dict или tensor)
            deterministic: Если True, использовать детерминированное действие
            
        Returns:
            action, log_prob
        """
        if deterministic:
            # Для детерминированного действия берём argmax
            # if isinstance(state, dict):
            #     state = self.translator.translate(state)
            
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            move_logits, kick_logit, _ = self.forward(state)
            
            # Детерминированный выбор
            move = (move_logits > 0).float()
            kick = (kick_logit > 0).float()
            
            action = torch.cat([move, kick], dim=-1)
            
            # Вычисляем log_prob для детерминированного действия
            move_dist = torch.distributions.Bernoulli(logits=move_logits)
            kick_dist = torch.distributions.Bernoulli(logits=kick_logit)
            logp = move_dist.log_prob(move).sum(-1) + kick_dist.log_prob(kick).squeeze(-1)
            
            if action.size(0) == 1:
                return action.squeeze(0), logp.squeeze(0)
            return action, logp
        else:
            # Стохастический выбор - используем get_action
            result = self.get_action(state)
            return result[0], result[1]  # action, log_prob
