import numpy as np
import torch
import torch.nn.functional as F
from AI.Maksigma_net import Maksigma_net
from Draw.Reward_plotter import Reward_plotter
from AI.Training.Memory import Memory
import Constants

scaler = torch.amp.grad_scaler.GradScaler()


class PPO:
    def __init__(self, net=None):

        if(net == None):
            self.policy = Maksigma_net()
            self.policy_old = Maksigma_net()
        else:
            self.policy = net
            self.policy_old = net
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.5e-3)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.5e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.98)

        self.K_epochs = 10  # Количество эпох оптимизации
        self.eps_clip = 0.2  # Clipping parameter

        self.plotter = Reward_plotter()
    
    def select_action(self, state):
        # Используем старую политику для сбора данных
        with torch.no_grad():
            state = state.unsqueeze(0)
            action, log_prob, _, _ = self.policy_old.get_action(state)
        
        return action, log_prob  # Сохраняем log_prob
    
    def compute_returns_and_advantages(self, rewards, values, terminals, gamma=0.99, lam=0.95):
        returns = []
        advantages = []
        gae = 0
        next_value = 0

        for step in reversed(range(len(rewards))):
            mask = 1.0 - terminals[step]
            delta = rewards[step] + gamma * next_value * mask - values[step]
            gae = delta + gamma * lam * mask * gae
            advantages.insert(0, gae)
            next_value = values[step]
            returns.insert(0, gae + values[step])

        return torch.tensor(returns, device=Constants.device), torch.tensor(advantages, device=Constants.device)
    
    def update(self, memory: Memory, ep, logging=True):
        # Перевод наград и terminals в тензоры
        batch_rewards = memory.rewards
        batch_states = memory.states
        batch_actions = memory.actions
        batch_logps = memory.old_log_probs
        batch_terminals = memory.is_terminals

        # Получаем current value оценку с помощью критика
        with torch.no_grad():
            _, batch_values, _ = self.policy.evaluate_actions(batch_states, batch_actions)

        # Вычисляем returns и advantages по батчу
        returns, advantages = self.compute_returns_and_advantages(batch_rewards, batch_values.squeeze(-1), batch_terminals)

        # Нормализация advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(self.K_epochs):
            # with torch.amp.autocast_mode.autocast(Constants.device.type):
                new_log_probs, values, entropy = self.policy.evaluate_actions(batch_states, batch_actions)

                ratios = torch.exp(new_log_probs - batch_logps)
                mean_ratio = ratios.mean().item()
                clipped_fraction = ((ratios < 0.8) | (ratios > 1.2)).float().mean()

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values.squeeze(-1), returns)
                entropy_loss = entropy.mean()

                entropy_bonus = self.get_entropy_coef(ep) * entropy_loss
                loss = actor_loss + 0.5 * critic_loss - entropy_bonus

                self.optimizer.zero_grad(set_to_none=True)

                loss.backward()
                self.optimizer.step()

                self.lr_scheduler.step()
                if logging:
                    print(f"Epoch {epoch}: mean_ratio={mean_ratio:.4f}, "
                        f"clipped={clipped_fraction:.2%}, new_log_prob = {new_log_probs.mean().item():.4f}")

        if logging:
            self.plotter.update(ep, (returns - batch_values.detach().squeeze(-1)).mean().detach().item())

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.to(Constants.device)

    def get_entropy_coef(self, episode_num):
        start_coef = 0.07      # высокий exploration на старте
        end_coef = 0.07    # низкий бонус после долгого обучения
        decay_episodes = 100
        
        # Линейная интерполяция
        progress = min(episode_num / decay_episodes, 1.0)
        coef = start_coef * (1 - progress) + end_coef * progress
        return coef
        
    
    
    



