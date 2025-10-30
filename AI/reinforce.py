from matplotlib import pyplot as plt
import numpy as np
import pygame
import torch
from AI.Environment import Environment
from AI.Maksigma_net import Maksigma_net
from AI.Reward_plotter import Reward_plotter
from Draw.Drawing import Drawing
import matplotlib
# matplotlib.use('Agg')

class reinforce:
    def __init__(self, policy : Maksigma_net, lr, loss):
        self.current_ep = 0
        self.total_ep = 0
        self.lr = lr
        self.optim = torch.optim.Adam(policy.parameters(), lr=lr)
        self.prev_loss = loss
        self.plotter = Reward_plotter(window_size=100)

    def run_episode(self, env : Environment, policy : Maksigma_net, device="cpu"):
        s = torch.as_tensor(env.reset())
        traj = {"logps": [], "rewards": [], "entropies": []}
        done = False
        draw = Drawing(env.map)
        while not done:
            logits = policy(s)
            a, logp, ent = env.ai_action.translator.translate_output(s)
            # a, logp, ent, vel, kick = sample_action_and_stats(logits)
            # with torch.no_grad():
            ns, r, done = env.step(a.cpu())
            # pygame.time.delay(1000)
            draw.draw()
            pygame.display.update()
            traj["logps"].append(logp)
            traj["rewards"].append(torch.as_tensor(r, dtype=torch.float32))
            traj["entropies"].append(ent)
            s = torch.as_tensor(ns, dtype=torch.float32)
        return traj

    def compute_returns(self, rewards, gamma=0.99):
        G = 0.0
        out = []
        for r in reversed(rewards):
            G = r + gamma * G
            out.append(G)
        out.reverse()
        return torch.stack(out)

    def train(self, policy : Maksigma_net, env, episodes=1000, batch_episodes=4, gamma=0.995, alpha = 0.01, device="cpu"):
        # policy = policy().to(device)
        self.total_ep = episodes
        optim = self.optim
        ep = 0
        total_loss = 0.0; total_return = 0.0

        # print("WEIGHTS AFTER:\n\n\n\n")
        # with torch.no_grad():
        #     for name, p in policy.named_parameters():
        #         if "weight" in name:
        #             print(f"{name}: {tuple(p.shape)}")
        #             print(p.detach().cpu())

        # loss = torch.rand(100, requires_grad=True)
        # optim.zero_grad()
        # loss.backward()
        # optim.step()

        # print("WEIGHTS AFTER:\n\n\n\n")
        # with torch.no_grad():
        #     for name, p in policy.named_parameters():
        #         if "weight" in name:
        #             print(f"{name}: {tuple(p.shape)}")
        #             print(p.detach().cpu())
        initial_lr = self.lr
        buffer = []

        old_log_prob = -3

        ratios = []
        while ep < episodes:
            # аккумулируем несколько эпизодов, затем один апдейт
            traj = self.run_episode(env, policy, device)
            ep += 1
            self.current_ep = ep

            buffer.append(traj)
            if len(buffer) == batch_episodes:
                # Считаем loss по всем эпизодам
                all_logps, all_adv, all_entropy = [], [], []
            
                # Склеиваем в один батч
                for t in buffer:
                    # stats = update_policy(policy, optim, t, gamma, entropy_coef=0.1)
                    returns = self.compute_returns(t["rewards"], gamma)
                    adv = (returns - returns.mean()) / (returns.std() + 1e-8) # baseline на каждом эпизоде
                    # adv = (returns)# baseline на каждом эпизоде
                    all_logps.append(torch.stack(t["logps"]))
                    all_adv.append(adv)
                buffer = []
                logps_batch = torch.cat(all_logps)
                adv_batch = torch.cat(all_adv)

                adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

                l2_lambda = 1e-4
                l2_norm = sum(p.square().sum() for p in policy.parameters())

                if "entropies" in t:
                    all_entropy.append(torch.stack(t["entropies"]))
                entropy = torch.cat(all_entropy).mean()
                # entropy = torch.cat(all_entropy).mean()
                # loss = -(logps_batch * adv_batch).mean() - alpha * entropy
                loss = -(logps_batch * adv_batch).mean()

                optim.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1)
                print("gradient:\n")
                grad_stats(policy)
                optim.step()

                # print("WEIGHTS AFTER:\n\n\n\n")
                # with torch.no_grad():
                #     for name, p in policy.named_parameters():
                #         if "weight" in name:

                # Средние значения за батч
                mean_log_prob = logps_batch.mean()  # Должно расти (становиться менее отрицательным)
                ratio = torch.exp(mean_log_prob - old_log_prob)
                old_log_prob = mean_log_prob
                ratios.append(ratio.detach().numpy())
                mean_ratio = np.array(ratios).mean()         # Должно быть близко к 1.0
                # clipping_fraction = (ratios > 1.2).mean()  # Какая доля ratio была обрезана
                
                print("logp stats:")
                print(f"mean_log_prob = {mean_log_prob}, mean_ratio = {mean_ratio}")


                #             print(f"{name}: {tuple(p.shape)}")
                #             print(p.detach().cpu())

                buffer = []
                total_return += adv_batch.mean().item()
                summ = 0
                for i in range(len(traj["rewards"])):
                    summ += traj["rewards"][i]

                summ = 0
                for i in range(len(traj["rewards"])):
                        summ += traj["rewards"][i]
                print(f"ep={ep} loss={loss} total_return={total_return} total_rewards = {summ}")

                self.plotter.update(ep, loss.detach().item())
                # Показать график до закрытия окна
                # plt.ioff()
                plt.show()

                print(f"ep={ep} entropy_bonus = {alpha * entropy}")
                if ep % 200 == 0:
                    torch.save({
                        'episode': ep,
                        'model_state_dict': policy.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'loss': loss
                    }, f'checkpoint_ep{ep}_{policy.name}.pth')

        return policy
    
def sample_action_and_stats(logits):
        # logits теперь трактуем как параметры нормального распределения
        # Первые половина значений - это средние (mu), остальные - log(sigma)
        # Извлекаем параметры для непрерывного действия (velocity)
        mu = logits[:2]  # НЕ применяем sigmoid, пусть модель сама учится
        log_sigma = logits[2:4]
    
        # Ограничиваем sigma для стабильности
        sigma = torch.exp(log_sigma).clamp(min=0.3, max=2.0)
        # sigma = torch.ones_like(mu)
        
        # Извлекаем логит для дискретного действия (kick)
        kick_logit = logits[4]
        
        # ========== VELOCITY (Continuous) ==========
        # Создаём распределение для скорости
        velocity_dist = torch.distributions.Normal(mu, sigma)
        
        # Семплируем действие (rsample для градиентов)
        velocity_raw = velocity_dist.rsample()
        
        # Вычисляем log_prob ДО tanh (важно!)
        velocity_logp = velocity_dist.log_prob(velocity_raw).sum(-1)
        velocity_entropy = velocity_dist.entropy().sum(-1)
        
        # Применяем tanh для ограничения ПОСЛЕ вычисления log_prob
        velocity_action = torch.tanh(velocity_raw)
        
        # ========== KICK (Discrete Binary) ==========
        # Создаём Bernoulli распределение для удара
        kick_dist = torch.distributions.Bernoulli(logits=kick_logit)
        
        # Семплируем kick (0 или 1)
        kick_action = kick_dist.sample()
        
        # Вычисляем log_prob и entropy для kick
        kick_logp = kick_dist.log_prob(kick_action)
        kick_entropy = kick_dist.entropy()
        
        # ========== ОБЪЕДИНЯЕМ ==========
        # Полное действие: [velocity_x, velocity_y, kick]
        action = torch.cat([velocity_action, kick_action.unsqueeze(-1)])
        
        # Суммарная log probability (независимые действия)
        total_logp = velocity_logp + kick_logp
        
        # Суммарная энтропия
        total_entropy = velocity_entropy + kick_entropy
        
        return action, total_logp, total_entropy

def grad_stats(model):
    with torch.no_grad():
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                # param.grad = torch.clip(param.grad, 0.1, 1)
                grads.append(param.grad.detach().abs().view(-1))  # берём по модулю
        if len(grads) == 0:
            print("Нет ненулевых градиентов.")
            return
        all_grads = torch.cat(grads)
        print(f"grad min: {all_grads.min():.6e}, grad max: {all_grads.max():.6e}")
        print(f"grad mean: {all_grads.mean():.6e}, grad std: {all_grads.std():.6e}")
