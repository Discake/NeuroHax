import numpy as np
import torch
import torch.nn.functional as F
from AI.Training.Memory import Memory
import Constants

class PPO:
    def __init__(self, net=None, net_old=None, episode_offset=0, episodes_total=500):
        # Если net_old не передан, создаём копию net
        # Для моделей, требующих аргументы (как SimpleModel), нужно передавать net_old явно

        self.episode_offset = episode_offset
        self.episodes_total = episodes_total
        self.policy = net
        if net_old is not None:
            self.policy_old = net_old
        else:
            # Пытаемся создать копию, обрабатывая случай когда нужны аргументы
            try:
                self.policy_old = net.__class__()
            except TypeError:
                # Если не получается (как для SimpleModel), создаём shallow copy
                import copy
                self.policy_old = copy.deepcopy(net)
        
        # Синхронизируем веса
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.vf_coef = 0.5  # увеличено для более сильного обучения критика (было 0.2)

        # === ОПТИМИЗАТОР ДЛЯ ВСЕЙ СЕТИ (PPO-эпохи) ===
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3, eps=1e-8)

        # === ОПТИМИЗАТОР ТОЛЬКО ДЛЯ КРИТИКА (value warmup) ===
        # Отдельный оптимизатор гарантирует, что actor не трогается во время warmup
        self.critic_optimizer = torch.optim.Adam(self.policy.critic_parameters(), lr=1e-3, eps=1e-8)

        # === ЕДИНЫЙ ПЛАНИРОВЩИК ===
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1, # Снижаем до 10% от начального
            total_iters=1000 # Более плавное снижение на протяжении большего числа эпизодов
        )
        
        # === АДАПТИВНЫЕ ЭПОХИ ===
        # ratio_std к epoch 4 достигает лишь 0.04–0.15 → политика слабо меняется за 5 эпох.
        # Critic отдельный → безопасно увеличить без риска для actor warmup.
        self.K_epochs_initial = 12
        self.K_epochs_final = 8

        self.eps_clip = 0.2

        # === БЕГУЩАЯ EMA СТАНДАРТНОГО ОТКЛОНЕНИЯ НАГРАД ===
        # Предотвращает скачки reward_scale между эпизодами (нет голов → std=6, есть голы → std=200)
        self._reward_scale_ema = None
        self._reward_scale_alpha = 0.05  # медленное обновление (≈ окно 20 эпизодов)

        # === ЭНТРОПИЯ ДЛЯ ИССЛЕДОВАНИЯ ===
        # ВАЖНО: max entropy для 5 Bernoulli = 5*ln(2) = 3.47 нат.
        # При coef=0.1 энтропийный бонус ≈ 0.35, что в 6–30× больше actor_loss (~0.01–0.05).
        # Следствие: политика застревает на 99.9% max entropy → actor gradient не работает.
        # Снижаем до coef=0.01 чтобы бонус (≈0.035) был сопоставим с actor_loss, а не доминировал.
        self.entropy_coef_initial = 0.007  # для fine-tuning зрелой модели: уточнение, а не исследование
        self.entropy_coef_final = 0.001
        self.entropy_decay_start = 20
        # Эффективная точка старта спада — сбрасывается при обновлении лаг-оппонента
        self._entropy_ep_offset = 0

    def get_entropy_coef(self, episode):
        """Энтропия для исследования с последующим снижением.
        При вызове bump_entropy_on_lag_update() спад перезапускается от текущего эпизода.
        """
        ep = (episode - self.episode_offset) - self._entropy_ep_offset
        if ep < self.entropy_decay_start:
            return self.entropy_coef_initial

        # Линейное снижение после entropy_decay_start
        progress = (ep - self.entropy_decay_start) / (self.episodes_total - self.entropy_decay_start)
        progress = min(progress, 1.0)

        coef = self.entropy_coef_initial * (1 - progress) + self.entropy_coef_final * progress
        return coef

    def bump_entropy_on_lag_update(self, episode):
        """Вызывать при обновлении lagged-оппонента.
        Перезапускает спад энтропии от текущего эпизода — агент снова получает
        достаточно исследования для адаптации к обновлённому оппоненту.
        """
        self._entropy_ep_offset = episode - self.episode_offset
        print(f"  [entropy reset at ep {episode}: coef → {self.entropy_coef_initial:.4f}]")


    def get_K_epochs(self, episode):
        """Уменьшаем эпохи со временем"""
        ep = episode - self.episode_offset
        progress = ep / self.episodes_total
        K = int(self.K_epochs_initial - progress * (self.K_epochs_initial - self.K_epochs_final))
        return max(K, self.K_epochs_final)
    
    def compute_returns_and_advantages(self, rewards, values, terminals, truncated, 
                                   gamma=0.99, lam=0.96):
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

    def update_combined(self, memory: Memory, ep, minibatch_size=1000, label="",
                        update_actor=True, step_scheduler=True):

        self.current_ep = ep

        # === КРИТИЧЕСКАЯ ДИАГНОСТИКА ===
        tag = f" [{label}]" if label else ""
        print(f"\n{'='*60}")
        print(f"EPISODE {ep} DIAGNOSTICS{tag}")
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

        # Нормализуем награды ПЕРЕД GAE чтобы value network и rewards были в одном масштабе.
        # Делим только на std (сохраняем знак и направление сигналов), не на mean.
        rewards_arr = np.array(memory.rewards, dtype=np.float32)
        batch_std = float(rewards_arr.std())
        # EMA сглаживает скачки scale между "голевыми" (std≈200) и "безголевыми" (std≈6) эпизодами
        if self._reward_scale_ema is None:
            self._reward_scale_ema = max(batch_std, 1.0)
        else:
            self._reward_scale_ema = (1 - self._reward_scale_alpha) * self._reward_scale_ema \
                                     + self._reward_scale_alpha * max(batch_std, 1.0)
        # Берём max(EMA, batch_std): EMA сглаживает, но никогда не занижает текущий масштаб
        # Занижение (EMA < batch_std) → value_loss=1000+ → катастрофа на старте обучения
        reward_scale = max(self._reward_scale_ema, batch_std, 1.0)
        scaled_rewards = [r / reward_scale for r in memory.rewards]

        returns, advantages = self.compute_returns_and_advantages(
            scaled_rewards, batch_values, memory.is_terminals, memory.is_truncated,
            gamma=0.99, lam=0.97)

        # Нормализация advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Статистика
        print(f"reward_scale={reward_scale:.1f} | raw_rewards: mean={rewards_arr.mean():.1f}, std={rewards_arr.std():.1f}")
        print(f"scaled returns: mean={returns.mean():.3f}, std={returns.std():.3f}")
        print(f"values: mean={batch_values.mean():.3f}, std={batch_values.std():.3f}")

        # === ФАЗА 1: РАЗОГРЕВ VALUE FUNCTION ===
        # Только critic_backbone + value_head обновляются.
        # Actor backbone заморожен → ratio_std = 0 при начале PPO-эпох.
        VALUE_WARMUP_EPOCHS = 16
        for _ in range(VALUE_WARMUP_EPOCHS):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                end = min(start + minibatch_size, batch_size)
                mb_idx = indices[start:end]
                v = self.policy.get_value(batch_states[mb_idx]).reshape(-1)
                vf_loss = F.mse_loss(v, returns[mb_idx])
                self.critic_optimizer.zero_grad()
                (self.vf_coef * vf_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.policy.critic_parameters(), max_norm=0.5)
                self.critic_optimizer.step()

        # Пересчитываем advantages с разогретой value function
        with torch.no_grad():
            warmed_values = self.policy.get_value(batch_states).squeeze()
        returns, advantages = self.compute_returns_and_advantages(
            scaled_rewards, warmed_values, memory.is_terminals, memory.is_truncated,
            gamma=0.991, lam=0.96)
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        print(f"after warmup: values mean={warmed_values.mean():.3f}, std={warmed_values.std():.3f}")

        # === ФАЗА 2: СТАНДАРТНЫЕ PPO-ЭПОХИ (actor + critic) ===
        if update_actor:
            K_epochs = self.get_K_epochs(ep)
            entropy_coef = self.get_entropy_coef(ep)
            for epoch in range(K_epochs):
                indices = torch.randperm(batch_size)

                for start in range(0, batch_size, minibatch_size):
                    end = min(start + minibatch_size, batch_size)
                    mb_indices = indices[start:end]

                    mb_states = batch_states[mb_indices]
                    mb_actions_final = batch_actions_final[mb_indices]
                    mb_old_logps = batch_logps[mb_indices]
                    mb_advantages = advantages[mb_indices]
                    mb_returns = returns[mb_indices]

                    new_log_probs, _, entropy = self.policy.evaluate_actions(mb_states, mb_actions_final)
                    values = self.policy.get_value(mb_states).reshape(-1)

                    ratios = torch.exp(new_log_probs - mb_old_logps)
                    surr1 = ratios * mb_advantages
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(values, mb_returns)
                    entropy_bonus = entropy.mean()
                    total_loss = actor_loss + self.vf_coef * value_loss - entropy_coef * entropy_bonus

                    self.optimizer.zero_grad()
                    total_loss.backward()

                    if start == 0:
                        print(f"  epoch {epoch}: value={value_loss.item():.3f}, actor={actor_loss.item():.4f}, "
                              f"entropy={entropy_coef * entropy_bonus.item():.4f}, "
                              f"ratio_mean={ratios.mean().item():.3f}, ratio_std={ratios.std().item():.4f}")
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                    self.optimizer.step()

            print(f"Entropy coefficient: {entropy_coef:.4f}")
            print(f"K epochs used: {K_epochs}")
        else:
            print(f"  [actor update skipped — critic only]")

        # Обновление планировщика и старой политики
        if step_scheduler:
            self.scheduler.step()
        self.policy_old.load_state_dict(self.policy.state_dict())

        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}\n")



