import numpy as np
import torch
import torch.nn.functional as F
from Draw.Reward_plotter import Reward_plotter
from AI.Training.Memory import Memory
import Constants

scaler = torch.amp.grad_scaler.GradScaler()


class PPO:
    def __init__(self, net=None):

        self.policy = net

        # === КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Создаем ОТДЕЛЬНУЮ старую политику ===
        if hasattr(net, '__class__'):
            self.policy_old = net.__class__()  # Создаем новый экземпляр того же класса
        else:
            # Fallback - клонируем через deepcopy
            import copy
            self.policy_old = copy.deepcopy(net)
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.5e-3)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.98)

        self.K_epochs = 3   # Меньше эпох!
        self.eps_clip = 0.2

        self.plotter = Reward_plotter()
    
    def compute_returns_and_advantages(self, rewards, values, terminals, gamma=0.995, lam=0.95):
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
    

    def update_with_minibatches_default(self, memory: Memory, ep, minibatch_size=4096):

        # === ИСПРАВЛЕННАЯ нормализация наград ===
        rewards_tensor = torch.tensor(memory.rewards, dtype=torch.float32)
        normalized_rewards = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        normalized_rewards = torch.clamp(normalized_rewards, -5, 5)
        memory.rewards = normalized_rewards.tolist()
        



        # Подготовка данных
        batch_size = len(memory.states)
        print(f"Breaking batch {batch_size} into minibatches of {minibatch_size}")
        
        batch_states = torch.stack([torch.tensor(s) for s in memory.states])
        batch_actions_final = torch.stack([torch.tensor(a) for a in memory.actions_final])
        # batch_actions_raw = torch.stack([torch.tensor(a) for a in memory.actions_raw])
        batch_logps = torch.tensor(memory.old_log_probs)
        
        # Используем СТАРУЮ политику для базового value
        with torch.no_grad():
            _, batch_values, _ = self.policy_old.evaluate_actions(
                batch_states, batch_actions_final)
        
        returns, advantages = self.compute_returns_and_advantages(
            memory.rewards, batch_values.squeeze(-1), memory.is_terminals)
        
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.plotter.update(ep, advantages.numpy().max())
        
        # Mini-batch training
        indices = torch.randperm(batch_size)
        
        total_losses = []
        for epoch in range(self.K_epochs):
            total_clips = 0
            total_kl = 0
            num_minibatches = 0
            
            for start in range(0, batch_size, minibatch_size):
                end = min(start + minibatch_size, batch_size)
                mb_indices = indices[start:end]
                
                mb_states = batch_states[mb_indices]
                mb_actions_final = batch_actions_final[mb_indices]
                # mb_actions_raw = batch_actions_raw[mb_indices]
                mb_old_logps = batch_logps[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                clip_frac, kl, loss = self._update_minibatch_default(
                    mb_states, mb_actions_final, mb_old_logps, mb_advantages, mb_returns)
                
                total_losses.append(loss.detach().item())

                total_clips += clip_frac
                total_kl += kl
                num_minibatches += 1
                
                # Early stopping на уровне минибатча
                if clip_frac > 0.4 or kl > 0.01:
                    print(f"⚠️  Stopping epoch {epoch} early due to instability")
                    break
            
            avg_clip = total_clips / num_minibatches if num_minibatches > 0 else 0
            avg_kl = total_kl / num_minibatches if num_minibatches > 0 else 0
            
            print(f"Epoch {epoch}: avg_clip={avg_clip:.2%}, avg_kl={avg_kl:.6f}")
            
            if avg_clip > 0.4 or avg_kl > 0.01:
                print("⚠️⚠️  High average clipping/KL, stopping early")
                break

        # Обновляем старую политику
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.lr_scheduler.step()

    def _update_minibatch_default(self, states, actions_final, old_logps, advantages, returns):
        """Обновление на одном мини-батче"""
        # ИСПОЛЬЗУЕМ actions_raw для корректного log_prob
        new_log_probs, values, entropy = self.policy.evaluate_actions(
                states, actions_final)
        
        ratios = torch.exp(new_log_probs - old_logps)
        clipped_ratios = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
        clipped_fraction = (ratios != clipped_ratios).float().mean().item()
        kl_div = (old_logps - new_log_probs).mean().item()
        
        # Loss computation
        surr1 = ratios * advantages
        surr2 = clipped_ratios * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(values.squeeze(-1), returns)
        
        loss = actor_loss + 0.5 * critic_loss
            
        total_loss = loss - 0.07 * entropy.mean()
        
        # Градиентный шаг
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.1)
        self.optimizer.step()
        
        # print(reg_info)
        
        return clipped_fraction, kl_div, total_loss
    
    



