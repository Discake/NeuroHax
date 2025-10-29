import numpy as np
import torch
import torch.nn.functional as F
from AI.Maksigma_net import Maksigma_net
from AI.Reward_plotter import Reward_plotter
from AI.memory import Memory
import Constants

class PPO:
    def __init__(self, net=None):

        if(net == None):
            self.policy = Maksigma_net()
            self.policy_old = Maksigma_net()
        else:
            self.policy = net
            self.policy_old = net
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.5e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.98)

        self.K_epochs = 200  # Количество эпох оптимизации
        self.eps_clip = 0.2  # Clipping parameter

        self.plotter = Reward_plotter()
    
    def select_action(self, state):
        # Используем старую политику для сбора данных
        with torch.no_grad():
            state = torch.tensor(state, device=Constants.device).unsqueeze(0)
            action, log_prob, _, _ = self.policy_old.get_action(state)
        
        return action, log_prob  # Сохраняем log_prob
    
    def compute_returns(self, rewards, gamma=1):
        """
        Args:
            rewards: list of floats (награда за каждый timestep в эпизоде)
            gamma: дисконтирующий множитель
        
        Returns:
            returns: список float, такая же длина, где returns[t] = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
        """
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.append(R)  # prepend
        returns.reverse()
        return returns
    
    def update(self, memory : Memory, ep, logging = False):
        # Данные из траекторий
        states = torch.stack([s.detach() for s in memory.states])
        actions = torch.stack([s.detach() for s in memory.actions])
        old_log_probs = torch.tensor(memory.old_log_probs, device=Constants.device)  # От policy_old
        rewards = memory.rewards
        
        # Вычисляем returns
        returns = self.compute_returns(rewards)
        returns = torch.tensor(returns, device=Constants.device)
        
        # K эпох оптимизации
        for epoch in range(self.K_epochs):
            # Вычисляем НОВЫЕ log_prob с текущей политикой
            new_log_probs, values, entropy = self.policy.evaluate_actions(
                states, actions
            )
            
            # ПРАВИЛЬНОЕ вычисление ratio (поэлементно!)
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Мониторинг
            mean_ratio = ratios.mean().item()
            clipped_fraction = ((ratios < 0.8) | (ratios > 1.2)).float().mean()
            
            
            # Advantages
            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = F.mse_loss(values, returns)
            entropy_loss = entropy.mean()

            entropy_bonus = self.get_entropy_coef(ep) * entropy_loss
            
            loss = actor_loss + 0.5 * critic_loss - entropy_bonus
            
            # Обновляем policy (не policy_old!)
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
            self.optimizer.step()
            self.lr_scheduler.step()
            if logging:
                print(f"Epoch {epoch}: mean_ratio={mean_ratio:.4f}, "
                    f"clipped={clipped_fraction:.2%}, new_log_prob = {new_log_probs.mean().item():.4f}")
                print('Current learning rate:', self.optimizer.param_groups[0]['lr'])

                self.plotter.update(ep, (returns - values.detach()).mean().detach().item())
        
        # После K эпох: обновляем старую политику
        self.policy_old.load_state_dict(self.policy.state_dict())

    def get_entropy_coef(self, episode_num):
        start_coef = 0.2      # высокий exploration на старте
        end_coef = 0.001      # низкий бонус после долгого обучения
        decay_episodes = 10000
        
        # Линейная интерполяция
        progress = min(episode_num / decay_episodes, 1.0)
        coef = start_coef * (1 - progress) + end_coef * progress
        return coef
        
    
    
    



