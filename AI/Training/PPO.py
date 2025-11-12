import numpy as np
import torch
import torch.nn.functional as F
from Draw.Reward_plotter import Reward_plotter
from AI.Training.Memory import Memory
import Constants

class RunningMeanStd:
    """Exponential Moving Average для returns"""
    def __init__(self, epsilon=1e-4, momentum=0.99):
        self.mean = 0.0
        self.var = 1.0
        self.momentum = momentum
        self.epsilon = epsilon
        self.initialized = False
    
    def update(self, x):
        if not self.initialized:
            self.mean = np.mean(x)
            self.var = np.var(x) + self.epsilon
            self.initialized = True
        else:
            batch_mean = np.mean(x)
            batch_var = np.var(x)
            
            self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            self.var = self.momentum * self.var + (1 - self.momentum) * batch_var
    
    def normalize(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + self.epsilon), -10, 10)

class PPO:
    def __init__(self, net=None):

        self.policy = net
        self.policy_old = net.__class__()
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # # === КРИТИЧЕСКАЯ ПРОВЕРКА ===
        # print("\n🔍 Network Architecture Check:")
        # print(f"Policy net parameters: {sum(p.numel() for p in self.policy.policy_net.parameters()):,}")
        # print(f"Value net parameters: {sum(p.numel() for p in self.policy.value_net.parameters()):,}")
        # print(f"Value head parameters: {sum(p.numel() for p in self.policy.value_head.parameters()):,}")
        
        # # Проверяем, что веса разные
        # policy_param_id = id(list(self.policy.policy_net.parameters())[0])
        # value_param_id = id(list(self.policy.value_net.parameters())[0])
        
        # if policy_param_id == value_param_id:
        #     print("🚨 CRITICAL: Policy and Value networks share parameters!")
        # else:
        #     print("✅ Policy and Value networks are separate")


        policy_lr = 3e-4

        # === ЗНАЧИТЕЛЬНО УВЕЛИЧЬТЕ POLICY LR ===
        self.policy_optimizer = torch.optim.Adam([
            {'params': self.policy.policy_net.parameters(), 'lr': policy_lr},  # Было 3e-4 или меньше
            {'params': self.policy.velocity_mean_head.parameters(), 'lr': policy_lr},
            {'params': self.policy.velocity_log_std_head.parameters(), 'lr': policy_lr},
            {'params': self.policy.kick_head.parameters(), 'lr': policy_lr},
        ], eps=1e-5)

        value_lr = 1e-3
        
        self.value_optimizer = torch.optim.Adam([
            {'params': self.policy.value_net.parameters(), 'lr': value_lr},  # Высокий lr!
            {'params': self.policy.value_head_1.parameters(), 'lr': value_lr},
            {'params': self.policy.value_head_2.parameters(), 'lr': value_lr},
        ], eps=1e-5)
        
        # === LR SCHEDULERS ===
        self.policy_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.policy_optimizer,
            start_factor=1.0,
            end_factor=0.3,  # Уменьшаем до 30% от начального
            total_iters=100
        )
        
        self.value_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.value_optimizer,
            start_factor=1.0,
            end_factor=0.5,
            total_iters=100
        )
        
        # === АДАПТИВНЫЕ ЭПОХИ ===
        self.K_epochs_initial = 7
        self.K_epochs_final = 3
        self.episodes_total = 100

        self.eps_clip = 0.2

        self.plotter = Reward_plotter()
        # === RUNNING STATS ДЛЯ RETURNS ===
        self.return_rms = RunningMeanStd(momentum=0.99)
        # === БОЛЕЕ АГРЕССИВНОЕ СНИЖЕНИЕ ===
        self.entropy_coef_initial = 0.005  # Уменьшите начальное
        self.entropy_coef_final = 0.0001   # Почти ноль в конце
        self.entropy_decay_start = 5      # Начинаем снижать с episode 30


        # """Режим fine-tuning после convergence"""
        
        # # Уменьшаем epochs
        # self.K_epochs = 2  # Было 3
        
        # # Уменьшаем LR ещё сильнее
        # for param_group in self.policy_optimizer.param_groups:
        #     param_group['lr'] *= 0.5  # Половина от текущего
        
        # for param_group in self.value_optimizer.param_groups:
        #     param_group['lr'] *= 0.5
        
        # # Минимальный entropy coef
        # self.entropy_coef_final = 0.00005
        
        # print("🎯 Fine-tuning mode activated!")
    
    def get_entropy_coef(self, episode):
        """Агрессивное снижение entropy"""
        if episode < self.entropy_decay_start:
            return self.entropy_coef_initial
        
        progress = (episode - self.entropy_decay_start) / (100 - self.entropy_decay_start)
        progress = min(progress, 1.0)
        
        # Экспоненциальное снижение (быстрее чем линейное)
        coef = self.entropy_coef_initial * (self.entropy_coef_final / self.entropy_coef_initial) ** progress
        
        # print(f"  Entropy coef: {coef:.6f}")
        return coef


    def get_K_epochs(self, episode):
        """Уменьшаем эпохи со временем"""
        progress = episode / self.episodes_total
        K = int(self.K_epochs_initial - progress * (self.K_epochs_initial - self.K_epochs_final))
        return max(K, self.K_epochs_final)
    
    def compute_returns_and_advantages(self, rewards, values, terminals, truncated, 
                                   gamma=0.999, lam=0.95):
        returns = []
        advantages = []
        gae = 0
        
        # === СТАНДАРТНАЯ GAE ===
        if len(terminals) > 0:
            if terminals[-1] == 0 or truncated[-1] == 1:
                next_value = values[-1].item() if torch.is_tensor(values[-1]) else values[-1]
            else:
                next_value = 0.0
        else:
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
        
        # === НОРМАЛИЗАЦИЯ RETURNS PER EPISODE ===
        # Группируем по эпизодам
        # episode_returns = []
        # current_episode = []
        
        # for i, (ret, term) in enumerate(zip(returns, terminals)):
        #     current_episode.append(ret.item())
            
        #     if term == 1:
        #         # Нормализуем эпизод
        #         if len(current_episode) > 1:
        #             ep_mean = np.mean(current_episode)
        #             ep_std = np.std(current_episode) + 1e-8
        #             normalized_ep = [(r - ep_mean) / ep_std for r in current_episode]
        #         else:
        #             normalized_ep = current_episode
                
        #         episode_returns.extend(normalized_ep)
        #         current_episode = []
        
        # # Добавляем остаток
        # if current_episode:
        #     if len(current_episode) > 1:
        #         ep_mean = np.mean(current_episode)
        #         ep_std = np.std(current_episode) + 1e-8
        #         normalized_ep = [(r - ep_mean) / ep_std for r in current_episode]
        #     else:
        #         normalized_ep = current_episode
        #     episode_returns.extend(normalized_ep)
        
        # returns = torch.tensor(episode_returns, dtype=torch.float32, device=Constants.device)
        
        # Мягкий clipping
        # returns = torch.clamp(returns, -50, 50)
        # advantages = torch.clamp(advantages, -15, 15)
        
        return returns, advantages

    

    def update_with_minibatches_default(self, memory: Memory, ep, minibatch_size=256):
        
        
        # === БОЛЕЕ ЖЁСТКИЙ CLIPPING ===
        # rewards_clipped = torch.clamp(memory.rewards, -50, 50)  # Было [-100, 100]!
        # memory.rewards = rewards_clipped

        self.current_ep = ep

        # === КРИТИЧЕСКАЯ ДИАГНОСТИКА ===
        print(f"\n{'='*60}")
        print(f"EPISODE {ep} DIAGNOSTICS")
        print(f"{'='*60}")


        # Подготовка данных
        batch_size = len(memory.states)
        print(f"Breaking batch {batch_size} into minibatches of {minibatch_size}")
        
        batch_states = torch.stack([torch.tensor(s) for s in memory.states])
        batch_actions_final = torch.stack([torch.tensor(a) for a in memory.actions_final])
        # batch_actions_raw = torch.stack([torch.tensor(a) for a in memory.actions_raw])
        batch_logps = torch.tensor(memory.old_log_probs)
        
        # === ИСПОЛЬЗУЙТЕ СТАРУЮ VALUE FUNCTION ===
        with torch.no_grad():
            _, _, _, batch_values = self.policy_old.forward(batch_states)  # policy_old!
        
        returns, advantages = self.compute_returns_and_advantages(
            memory.rewards, batch_values, memory.is_terminals, memory.is_truncated)
        
        # === ДОБАВЬТЕ ЭТОТ БЛОК ПРАВИЛЬНОЙ НОРМАЛИЗАЦИИ ===
        # Это стабилизирует цель для Value-сети
        # if returns.std() > 1e-8:
        #     returns = (returns - returns.mean()) / returns.std()
        # ====================================================
        
        # returns_normalized = self.return_rms.normalize(returns_np)
        # returns_normalized = torch.tensor(
        #     returns_normalized, 
        #     dtype=torch.float32,
        #     device=Constants.device
        # )
        
        # print(f"\n📊 Returns normalization (RMS):")
        # print(f"   Raw: mean={returns.mean():.2f}, std={returns.std():.2f}")
        # print(f"   RMS: mean={self.return_rms.mean:.2f}, std={np.sqrt(self.return_rms.var):.2f}")
        # print(f"   Normalized: [{returns_normalized.min():.2f}, {returns_normalized.max():.2f}]")






        # # === CLIPPING НАГРАД (перед GAE) ===      
        # rewards_tensor = torch.tensor(memory.rewards, dtype=torch.float32)

        
        # Вариант 2: Soft clipping (tanh)
        # rewards_clipped = torch.tanh(rewards_tensor / 100) * 100
        
        # print(f"Rewards before clip: [{rewards_tensor.min():.1f}, {rewards_tensor.max():.1f}]")
        # print(f"Rewards after clip:  [{rewards_clipped.min():.1f}, {rewards_clipped.max():.1f}]")
        
        # memory.rewards = rewards_clipped.tolist()
        

        raw_mean = advantages.mean().item()
        raw_std = advantages.std().item()
        raw_min = advantages.min().item()
        raw_max = advantages.max().item()
        
        print(f"\n📊 RAW Advantages (before normalization):")
        print(f"   Mean: {raw_mean:.3f}, Std: {raw_std:.3f}")
        print(f"   Range: [{raw_min:.3f}, {raw_max:.3f}]")

        # 2. Value function quality
        with torch.no_grad():
            _, _, _, current_values = self.policy.forward(batch_states)
            value_mse = F.mse_loss(current_values.squeeze(-1), returns).item()
            value_explained_var = 1 - (returns - current_values.squeeze(-1)).var() / returns.var()
            value_explained_var = value_explained_var.item()
        
            print(f"\n🎯 Value Function:")
            print(f"   MSE Loss: {value_mse:.4f}")
            print(f"   Explained Variance: {value_explained_var:.3f}")
            
            # Критический check
            if value_mse > 100:
                print(f"   🚨 CRITICAL: Value function MSE too high!")
            if value_explained_var < 0:
                print(f"   🚨 CRITICAL: Value function worse than predicting mean!")
            
            # 3. Returns and Values
            print(f"\n💰 Returns vs Values:")
            print(f"   Returns:     mean={returns.mean():.3f}, std={returns.std():.3f}")
            print(f"   Values:      mean={current_values.mean():.3f}, std={current_values.std():.3f}")
            print(f"   Return range: [{returns.min():.2f}, {returns.max():.2f}]")
            
            self.returns_mean = returns.mean().item()  # .item()!
            self.returns_std = returns.std().item()    # .item()!

            if self.returns_std < 1e-8:
                self.returns_std = 1.0  # Защита от деления на ноль

            # returns_normalized = (returns - self.returns_mean) / self.returns_std
            # 4. Rewards
            rewards_tensor = torch.tensor(memory.rewards)
            print(f"\n🎁 Rewards:")
            print(f"   Mean: {rewards_tensor.mean():.3f}, Std: {rewards_tensor.std():.3f}")
            print(f"   Range: [{rewards_tensor.min():.2f}, {rewards_tensor.max():.2f}]")
            print(f"   Positive: {(rewards_tensor > 0).sum()}/{len(rewards_tensor)} ({100*(rewards_tensor > 0).float().mean():.1f}%)")

        
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        # 5. После нормализации
        if raw_std > 1e-8:
            
            print(f"\n📏 After Normalization:")
            print(f"   Range: [{advantages.min():.3f}, {advantages.max():.3f}]")
            
            # Проверка - сколько будет обрезано
            will_clip = (advantages.abs() > 5).sum().item()
            print(f"   Will be clipped: {will_clip}/{len(advantages)} ({100*will_clip/len(advantages):.1f}%)")
            
            # advantages = torch.clamp(advantages, -10, 10)
            
            print(f"   After Clipping: [{advantages.min():.3f}, {advantages.max():.3f}]")
        else:
            print(f"\n🚨 CRITICAL: Advantage std too low ({raw_std:.6f}) - possible collapse!")
        
        # Проверка entropy
        with torch.no_grad():
            _, _, entropy = self.policy.evaluate_actions(batch_states, batch_actions_final)
            mean_entropy = entropy.mean().item()
        
        print(f"\n🎲 Policy Entropy: {mean_entropy:.4f}")
        if mean_entropy < 0.01:
            print(f"   🚨 WARNING: Very low entropy - policy becoming deterministic!")
        
        print(f"{'='*60}\n")






        self.plotter.update(ep, advantages.numpy().min())
        
        # Mini-batch training
        indices = torch.randperm(batch_size)
        
        K_epochs = self.get_K_epochs(ep)
        print(f"Using {K_epochs} epochs for this update")

        for epoch in range(K_epochs):
            total_clips = 0
            total_kl = 0
            num_minibatches = 0
            
            for start in range(0, batch_size, minibatch_size):
                end = min(start + minibatch_size, batch_size)
                mb_indices = indices[start:end]
                
                mb_states = batch_states[mb_indices]
                mb_actions_final = batch_actions_final[mb_indices]
                mb_old_logps = batch_logps[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                clip_frac, kl = self._update_policy_only(
                    mb_states, mb_actions_final, mb_old_logps, mb_advantages)
                
                total_clips += clip_frac
                total_kl += kl
                num_minibatches += 1
            
            avg_clip = total_clips / num_minibatches if num_minibatches > 0 else 0
            avg_kl = total_kl / num_minibatches if num_minibatches > 0 else 0
            
            print(f"Epoch {epoch}: avg_clip={avg_clip:.2%}, avg_kl={avg_kl:.6f}")
            

            # Проверяем entropy после каждой эпохи
            with torch.no_grad():
                _, _, entropy = self.policy.evaluate_actions(
                    batch_states[:2048], 
                    batch_actions_final[:2048]
                )
                current_entropy = entropy.mean().item()
            
            if avg_kl > 0.015:
                print("⚠️  High KL, stopping early")
                break
            # === EARLY STOPPING ПО ENTROPY ===
            if current_entropy > 2.8:
                print(f"⚠️  Entropy too high ({current_entropy:.4f}), stopping early")
                break

        # === ЗАТЕМ ОБНОВЛЯЕМ VALUE FUNCTION ===
        print("\n🎯 Updating Value Function...")
        self._update_value_function(batch_states, returns)


        # Обновляем learning rates
        self.policy_scheduler.step()
        self.value_scheduler.step()
        
        current_lr = self.policy_optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.6f}")
        
        # Обновляем старую политику
        self.policy_old.load_state_dict(self.policy.state_dict())

    def _update_policy_only(self, states, actions_final, old_logps, advantages):
        """Обновление только policy"""
        new_log_probs, _, entropy = self.policy.evaluate_actions(states, actions_final)
        
        ratios = torch.exp(new_log_probs - old_logps)
        clipped_ratios = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
        clipped_fraction = (ratios != clipped_ratios).float().mean().item()
        kl_div = (old_logps - new_log_probs).mean().item()
        
        surr1 = ratios * advantages
        surr2 = clipped_ratios * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        entropy_coef = self.get_entropy_coef(self.current_ep)
        total_loss = actor_loss - entropy_coef * entropy.mean()
        
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in self.policy_optimizer.param_groups for p in group['params']], 
            max_norm=0.5
        )
        self.policy_optimizer.step()
        
        return clipped_fraction, kl_div

    def _update_value_function(self, states, returns):
        """Обновление value с защитой от gradient explosion"""
        
        # Получаем предсказания СТАРОЙ сети
        with torch.no_grad():
            _, _, _, old_values = self.policy_old.forward(states)
            if old_values.dim() > 1:
                old_values = old_values.squeeze(-1)
        
        # === ДЕТЕКТОР GRADIENT EXPLOSION ===
        consecutive_high_grads = 0
        max_consecutive = 2
        
        for value_iter in range(20):
            _, _, _, values = self.policy.forward(states)
            
            if values.dim() > 1:
                values = values.squeeze(-1)
            
            _, _, _, values = self.policy.forward(states)
            values = values.squeeze(-1)

            # === НОВЫЙ РАСЧЕТ ПОТЕРЬ С КЛИППИНГОМ ===
            # Клиппинг предсказаний
            values_clipped = old_values + torch.clamp(
                values - old_values, -self.eps_clip, self.eps_clip
            )
            # Потери на оригинальных и клиппированных значениях
            loss_v1 = F.smooth_l1_loss(values, returns, reduction='none')
            loss_v2 = F.smooth_l1_loss(values_clipped, returns, reduction='none')
            # Итоговые потери - максимум из двух
            value_loss = torch.max(loss_v1, loss_v2).mean()
            # ==========================================

            self.value_optimizer.zero_grad()
            value_loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.policy.value_net.parameters()) + 
                list(self.policy.value_head_1.parameters()) +
                list(self.policy.value_head_2.parameters()),
                max_norm=2.0
            )
            
            # Gradient explosion detection
            if grad_norm > 50:
                consecutive_high_grads += 1
                if consecutive_high_grads >= max_consecutive:
                    print(f"  🚨 Gradient explosion! Stopping at iter {value_iter}")
                    break
            else:
                consecutive_high_grads = 0
            
            self.value_optimizer.step()
            
            if value_iter % 3 == 0:
                with torch.no_grad():
                    _, _, _, current_vals = self.policy.forward(states)
                    if current_vals.dim() > 1:
                        current_vals = current_vals.squeeze(-1)
                    
                    pred_std = current_vals.std().item()
                    target_std = returns.std().item()
                
                print(f"  Value iter {value_iter}: loss={value_loss.item():.2f}, "
                    f"grad_norm={grad_norm:.4f}, pred_std={pred_std:.2f}, target_std={target_std:.2f}")
            
            if value_loss.item() < 0.1:
                break
    
    



