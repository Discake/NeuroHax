"""
Дообучение — двухфазное:
  Фаза 1: против замороженной случайной модели → учимся забивать быстро
  Фаза 2: lagged self-play (задержка lag эпизодов)

Награды:
  - Все движенческие награды SimpleModelEnvironment сохранены
  - STEP_PENALTY: штраф за каждый шаг → стимул забивать как можно быстрее
  - Обучается только P1; данные P2 не собираются

Запуск:
    python finetune_goal_scoring.py --weights models/simple_model_vs.pth
    python finetune_goal_scoring.py --weights models/simple_model_vs.pth --phase1 80 --lag 40 --step-penalty 8.0
"""

import torch
import numpy as np
import multiprocessing as mp
import time
import os
import sys
import argparse
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from AI.Models.SimpleModel.Policy import SimpleModel
from AI.Models.SimpleModel.SimpleModelTranslator import SimpleModelTranslator
from AI.Models.SimpleModel.SimpleModelEnvironment import SimpleModelEnvironment
from AI.Training.PPO import PPO
from AI.Training.Memory import Memory
from AI.Training.ChunkedMmapBuffer import ChunkedMmapBuffer
from training_plotter import TrainingPlotter
from Core.Domain.Entities.Map import Map
import Constants


# ── Воркер ───────────────────────────────────────────────────────────────────

class FineTuneWorker:
    @staticmethod
    def run_worker(worker_id, max_steps, state_dict_1, state_dict_2,
                   shm_name, step_penalty, seed=None):
        try:
            if seed is not None:
                torch.manual_seed(seed + worker_id * 1000)
                np.random.seed(seed + worker_id * 1000)

            map_obj = Map()
            map_obj.load_random()

            translator1 = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
            translator2 = SimpleModelTranslator(map_obj, map_obj.players_team2[0], is_team_1=False)

            policy1 = SimpleModel(translator1)
            policy2 = SimpleModel(translator2)
            policy1.load_state_dict(state_dict_1)
            policy2.load_state_dict(state_dict_2)

            env = SimpleModelEnvironment(policy1, policy2, num_steps=max_steps)

            # Перебалансировка наград для fine-tuning:
            # движение сохранено (модель умеет бегать, не ломаем это),
            # но гол поднят настолько, чтобы стать доминирующим сигналом.
            # Баланс: ~100/step * 100 steps = 10k movement + 50k goal → гол = 83% сигнала.
            env.VELOCITY_TOWARD_BALL_REWARD = 2.0    # было 3.5; снижаем, не убираем
            env.DIRECTED_KICK_REWARD        = 6.0    # было 2.0; сильный сигнал точного удара
            env.GOAL_REWARD                 = 50000  # было 9000
            env.GOAL_CONCEDED_PENALTY       = -15000 # было -9000
            env.NEAR_BALL_REWARD            = 0.1
            env.BALL_IN_GOAL_ZONE           = 1.0

            state_size  = Constants.state_size
            action_size = 5
            step_size   = state_size + action_size + 4  # + reward, log_prob, terminal, truncated

            shm = ChunkedMmapBuffer()
            shm.open(max_steps, step_size, shm_name)

            s1, s2 = env.reset()
            p1_steps  = 0
            total_reward = 0
            goals_team1  = 0
            goals_team2  = 0
            done = False

            while p1_steps < max_steps and not done:
                with torch.no_grad():
                    action1, log_prob1 = policy1.select_action(s1, deterministic=False)
                    action2, _         = policy2.select_action(s2, deterministic=False)

                (ns1, ns2), (r1, _r2), done, info = env.step(action1, action2)

                r1_val = r1.item() - step_penalty  # штраф за каждый шаг

                if info.get('goal_team1', False):
                    goals_team1 += 1
                if info.get('goal_team2', False):
                    goals_team2 += 1

                total_reward += r1_val

                exp1 = np.concatenate([
                    s1.detach().flatten().numpy(),
                    action1.detach().flatten().numpy(),
                    [r1_val, log_prob1.item(),
                     1.0 if info.get('natural_done') else 0.0,
                     1.0 if info.get('truncated')    else 0.0],
                ])
                shm[p1_steps] = exp1
                p1_steps += 1
                s1, s2 = ns1, ns2

            shm.close(shm_name, delete_files=False, clear_files=False)
            return worker_id, total_reward, goals_team1, goals_team2

        except Exception as e:
            import traceback
            traceback.print_exc()
            return worker_id, -1000, 0, 0


# ── Сбор данных ──────────────────────────────────────────────────────────────

def collect(policy, opponent_sd, num_workers, max_steps, step_penalty, episode):
    state_size  = Constants.state_size
    action_size = 5
    step_size   = state_size + action_size + 4

    shm_names = [f"ft_exp_{i}.dat" for i in range(num_workers)]
    for name in shm_names:
        buf = ChunkedMmapBuffer()
        buf.create(name, max_steps, step_size)

    sd1 = policy.state_dict()

    args = [
        (i, max_steps, sd1, opponent_sd, shm_names[i], step_penalty, episode * 100 + i)
        for i in range(num_workers)
    ]

    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(FineTuneWorker.run_worker, args)

    all_s, all_a, all_lp, all_r, all_t, all_tr = [], [], [], [], [], []
    total_reward = goals_1 = goals_2 = 0

    for i, name in enumerate(shm_names):
        try:
            shm = ChunkedMmapBuffer()
            shm.open(max_steps, step_size, name)
            total_reward += results[i][1]
            goals_1      += results[i][2]
            goals_2      += results[i][3]

            for step in range(max_steps):
                row = shm[step]
                if np.all(row == 0):
                    break
                all_s.append(row[:state_size].tolist())
                all_a.append(row[state_size:state_size + action_size].tolist())
                all_r.append(float(row[state_size + action_size]))
                all_lp.append(float(row[state_size + action_size + 1]))
                all_t.append(float(row[state_size + action_size + 2]))
                all_tr.append(float(row[state_size + action_size + 3]))

            shm.close(name, delete_files=False, clear_files=True)
        except Exception as e:
            print(f"  [collect] error reading {name}: {e}")

    return all_s, all_a, all_lp, all_r, all_t, all_tr, total_reward, goals_1, goals_2


# ── Основной цикл ────────────────────────────────────────────────────────────

def finetune(weights_path: str, num_episodes: int = 300, num_workers: int = 6,
             max_steps: int = 512, save_interval: int = 10,
             save_path: str = "models/simple_model_ft.pth",
             phase1_episodes: int = 100, lag: int = 50,
             step_penalty: float = 5.0):

    phase2_start = phase1_episodes

    print(f"\n{'='*70}")
    print(f"  Дообучение — двухфазное (step_penalty={step_penalty:.1f}/шаг)")
    print(f"  Фаза 1 (ep 0–{phase1_episodes-1}):   случайный оппонент")
    print(f"  Фаза 2 (ep {phase1_episodes}–{num_episodes-1}): lagged self-play (lag={lag})")
    print(f"  Веса: {weights_path}  |  Воркеров: {num_workers}  |  Шагов: {max_steps}")
    print(f"{'='*70}\n")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Создаём модели
    map_obj    = Map()
    map_obj.load_random()
    translator1 = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
    translator2 = SimpleModelTranslator(map_obj, map_obj.players_team2[0], is_team_1=False)
    policy      = SimpleModel(translator1)
    policy_old  = SimpleModel(translator2)  # PPO baseline

    # Случайный оппонент (Фаза 1): захватываем ДО загрузки чекпоинта
    random_sd = {k: v.clone() for k, v in policy.state_dict().items()}

    # Загружаем существующие веса
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=Constants.device)
        policy.load_state_dict(state)
        policy_old.load_state_dict(state)
        print(f"  Веса загружены: {weights_path}\n")
    else:
        print(f"  [WARN] файл весов не найден, начинаем с нуля\n")

    # История весов для lagged self-play (Фаза 2)
    # Заполняем начальными весами — оппонент в начале Фазы 2 = модель из конца Фазы 1
    initial_sd = {k: v.clone() for k, v in policy.state_dict().items()}
    policy_history: deque = deque(
        ({k: v.clone() for k, v in initial_sd.items()} for _ in range(lag)),
        maxlen=lag,
    )

    # PPO с пониженным LR
    ppo = PPO(policy, policy_old)
    for pg in ppo.optimizer.param_groups:
        pg['lr'] = 5e-4
    ppo.scheduler.base_lrs = [5e-4 for _ in ppo.scheduler.base_lrs]

    plotter       = TrainingPlotter(save_dir=os.path.dirname(save_path) or ".",
                                    update_every=5)
    goals_history = []
    best_goals    = -1

    for episode in range(num_episodes):
        t0 = time.time()

        in_phase1 = episode < phase2_start
        phase_label = "rnd" if in_phase1 else f"lag{lag}"

        # Выбираем оппонента
        if in_phase1:
            opponent_sd = random_sd
        else:
            opponent_sd = policy_history[0]

        (states, actions, log_probs, rewards, terminals, truncated,
         total_reward, goals_1, goals_2) = collect(
            policy, opponent_sd, num_workers, max_steps, step_penalty, episode)

        # Обновляем историю ПОСЛЕ сбора (независимо от фазы — чтобы Фаза 2 начала с актуальных весов)
        policy_history.append({k: v.clone() for k, v in policy.state_dict().items()})

        if not states:
            print(f"Episode {episode+1}: нет данных")
            continue

        batch = len(states)

        mem = Memory()
        mem.states        = states
        mem.actions_final = actions
        mem.old_log_probs = log_probs
        mem.rewards       = rewards
        mem.is_terminals  = terminals
        mem.is_truncated  = truncated
        mem.copy_to_tensors()
        ppo.update_combined(mem, episode, minibatch_size=max(256, batch // 8), label="P1")

        policy_old.load_state_dict(policy.state_dict())

        avg_reward = np.mean(rewards) if rewards else 0
        elapsed    = time.time() - t0

        goals_history.append(goals_1 + goals_2)
        if len(goals_history) > 20:
            goals_history.pop(0)
        avg20 = np.mean(goals_history)

        plotter.update(episode + 1, goals_1, goals_2, 0, 0, avg_reward, total_reward, batch)

        if episode % 5 == 0 or episode == num_episodes - 1:
            print(f"Ep {episode+1:>4}/{num_episodes} [{phase_label}] | "
                  f"Goals {goals_1}-{goals_2} avg20={avg20:.1f} | "
                  f"Reward {avg_reward:.1f} | {elapsed:.1f}s")

        if episode % save_interval == 0 and episode > 0:
            torch.save(policy.state_dict(), save_path)
            print(f"  -> сохранено: {save_path}")

        total_goals = goals_1 + goals_2
        if total_goals > best_goals:
            best_goals = total_goals
            dir_name   = os.path.dirname(save_path)
            best_path  = os.path.join(dir_name, "best_" + os.path.basename(save_path))
            torch.save(policy.state_dict(), best_path)
            print(f"  -> BEST ({best_goals} goals): {best_path}")

    torch.save(policy.state_dict(), save_path)
    plotter.close()
    print(f"\n{'='*70}")
    print(f"Дообучение завершено. Модель: {save_path}")
    print(f"{'='*70}\n")


# ── Точка входа ──────────────────────────────────────────────────────────────

def main():
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Fine-tune goal scoring (2-phase)")
    parser.add_argument("--weights",       default="models/simple_model_vs.pth",
                        help="Путь к существующим весам")
    parser.add_argument("--episodes",      type=int,   default=300,
                        help="Всего эпизодов")
    parser.add_argument("--workers",       type=int,   default=6)
    parser.add_argument("--steps",         type=int,   default=512,
                        help="Шагов на воркера за эпизод")
    parser.add_argument("--save",          default="models/simple_model_ft.pth")
    parser.add_argument("--save-interval", type=int,   default=10)
    parser.add_argument("--phase1",        type=int,   default=60,
                        help="Эпизодов в Фазе 1 (случайный оппонент)")
    parser.add_argument("--lag",           type=int,   default=30,
                        help="Задержка оппонента в Фазе 2 (эпизодов)")
    parser.add_argument("--step-penalty",  type=float, default=80.0,
                        help="Штраф за каждый шаг (стимул быстрого гола)")
    args = parser.parse_args()

    finetune(
        weights_path    = args.weights,
        num_episodes    = args.episodes,
        num_workers     = args.workers,
        max_steps       = args.steps,
        save_interval   = args.save_interval,
        save_path       = args.save,
        phase1_episodes = args.phase1,
        lag             = args.lag,
        step_penalty    = args.step_penalty,
    )


if __name__ == "__main__":
    main()
