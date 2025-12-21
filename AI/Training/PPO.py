import numpy as np
import torch
import torch.nn.functional as F
from Draw.Reward_plotter import Reward_plotter
from AI.Training.Memory import Memory
import Constants

class PPO:
    def __init__(self, net=None):

        self.policy = net
        self.policy_old = net.__class__()
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.vf_coef = 0.5

        # === ЕДИНЫЙ ОПТИМИЗАТОР ДЛЯ ВСЕЙ СЕТИ ===
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4, eps=1e-5)
        
        # === ЕДИНЫЙ ПЛАНИРОВЩИК ===
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1, # Снижаем до 10% от начального
            total_iters=1000 # Более плавное снижение на протяжении большего числа эпизодов
        )
        
        # === АДАПТИВНЫЕ ЭПОХИ ===
        self.K_epochs_initial = 7
        self.K_epochs_final = 3
        self.episodes_total = 100

        self.eps_clip = 0.2

        self.plotter = Reward_plotter()
        # === БОЛЕЕ АГРЕССИВНОЕ СНИЖЕНИЕ ===
        self.entropy_coef_initial = 0.02  # Уменьшите начальное
        self.entropy_coef_final = 0.02  # Почти ноль в конце
        self.entropy_decay_start = 5      # Начинаем снижать с episode 30
    
    def get_entropy_coef(self, episode):
        """Агрессивное снижение entropy"""
        if episode < self.entropy_decay_start:
            return self.entropy_coef_initial
        
        progress = (episode - self.entropy_decay_start) / (100 - self.entropy_decay_start)
        progress = min(progress, 1.0)
        
        # Экспоненциальное снижение (быстрее чем линейное)
        coef = self.entropy_coef_initial * (self.entropy_coef_final / self.entropy_coef_initial) ** progress

        return coef


    def get_K_epochs(self, episode):
        """Уменьшаем эпохи со временем"""
        progress = episode / self.episodes_total
        K = int(self.K_epochs_initial - progress * (self.K_epochs_initial - self.K_epochs_final))
        return max(K, self.K_epochs_final)
    
    def compute_returns_and_advantages(self, rewards, values, terminals, truncated, 
                                   gamma=0.99, lam=0.95):
        returns = []
        advantages = []
        gae = 0
        
        # === СТАНДАРТНАЯ GAE ===
        # if len(terminals) > 0:
        #     if terminals[-1] == 0 or truncated[-1] == 1:
        #         next_value = values[-1].item() if torch.is_tensor(values[-1]) else values[-1]
        #     else:
        #         next_value = 0.0
        # else:
        #     next_value = 0.0
        
        next_value = 0.0
        
        for step in reversed(range(len(rewards))):
            if terminals[step] == 1 and truncated[step] == 0:
                mask = 0.0
            else:
                mask = 1.0
            
            current_value = values[step].item() if torch.is_tensor(values[step]) else values[step]
            
            delta = rewards[step] + gamma * next_value * mask - current_value
            gae = delta + gamma * lam * mask * gae
            
            advantages.insert(0, gae)
            next_value = current_value
            returns.insert(0, gae + current_value)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=Constants.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=Constants.device)
        
        return returns, advantages

    def update_combined(self, memory: Memory, ep, minibatch_size=1000):
        
        self.current_ep = ep

        # === КРИТИЧЕСКАЯ ДИАГНОСТИКА ===
        print(f"\n{'='*60}")
        print(f"EPISODE {ep} DIAGNOSTICS")
        print(f"{'='*60}")


        # Подготовка данных
        batch_size = len(memory.states)
        print(f"Breaking batch {batch_size} into minibatches of {minibatch_size}")
        
        batch_states = torch.stack([torch.tensor(s, device=Constants.device, dtype=torch.float32) for s in memory.states])
        batch_actions_final = torch.stack([torch.tensor(a, device=Constants.device, dtype=torch.float32) for a in memory.actions_final])
        batch_logps = torch.tensor(memory.old_log_probs, device=Constants.device, dtype=torch.float32)
        
        with torch.no_grad():
            # Рассчитываем returns и advantages ОДИН раз перед эпохами
            batch_values = self.policy_old.get_value(batch_states).squeeze()
        
        returns, advantages = self.compute_returns_and_advantages(
            memory.rewards, batch_values, memory.is_terminals, memory.is_truncated)
        
        # Нормализация advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Статистика
        values = self.policy_old.get_value(batch_states).squeeze()
        print(f"rewards: mean={memory.rewards.mean():.2f}, std={memory.rewards.std():.2f}")
        print(f"returns: mean={returns.mean():.2f}, std={returns.std():.2f}")
        print(f"values: mean={values.mean():.2f}, std={values.std():.2f}")

        # 2. Цикл обучения по эпохам и мини-батчам
        K_epochs = self.get_K_epochs(ep)
        for epoch in range(K_epochs):
            # Перемешиваем индексы на каждой эпохе
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                end = min(start + minibatch_size, batch_size)
                mb_indices = indices[start:end]
                
                # Извлекаем данные для мини-батча
                mb_states = batch_states[mb_indices]
                mb_actions_final = batch_actions_final[mb_indices]
                mb_old_logps = batch_logps[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # === КЛЮЧЕВОЙ МОМЕНТ: ВСЕ ПОТЕРИ РАССЧИТЫВАЮТСЯ ВМЕСТЕ ===
                
                # Получаем новые log_probs, предсказания value и энтропию от ТЕКУЩЕЙ политики
                new_log_probs, _, entropy = self.policy.evaluate_actions(mb_states, mb_actions_final)
                values = self.policy.get_value(mb_states).squeeze()
                
                # --- Policy Loss (Actor) ---
                ratios = torch.exp(new_log_probs - mb_old_logps)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # --- Value Loss (Critic) ---
                # Используем обычный MSE Loss. Ваш вариант с клиппингом тоже можно сюда вставить.
                value_loss = F.mse_loss(values, mb_returns)
                
                # --- Энтропия для исследования ---
                entropy_bonus = entropy.mean()
                
                # --- Комбинированная функция потерь ---
                entropy_coef = self.get_entropy_coef(ep)
                total_loss = actor_loss + self.vf_coef * value_loss - entropy_coef * entropy_bonus
                
                # --- Шаг оптимизации ---
                self.optimizer.zero_grad()
                total_loss.backward()

                if start == 0 and epoch == 0:  # Print only once per update
                    print(f"value loss: {value_loss.item():.4f}, actor loss: {actor_loss.item():.4f}, "
                        f"entropy bonus: {entropy_coef * entropy_bonus.item():.4f}, total loss: {total_loss.item():.4f}")
                # Клиппинг градиентов для всей сети
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
        
        # 3. Обновление планировщика и старой политики (после всех эпох)
        self.scheduler.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"Entropy coefficient: {entropy_coef:.4f}")
        print(f"K epochs used: {K_epochs}")
        print(f"{'='*60}\n")



