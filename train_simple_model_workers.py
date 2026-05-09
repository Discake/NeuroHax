"""
Скрипт для обучения SimpleModel подбегать к мячу
Использует multiprocessing воркеры для сбора данных

Запуск:
    python train_simple_model_workers.py
"""

import torch
import time
import multiprocessing as mp
import numpy as np
import sys
import os
import random


class _Tee:
    """Дублирует stdout в файл и в оригинальный поток одновременно."""
    def __init__(self, path, mode="a"):
        self._file   = open(path, mode, encoding="utf-8", buffering=1)
        self._stdout = sys.stdout
        sys.stdout   = self

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()

# Добавляем корень проекта в path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from AI.Models.SimpleModel.Policy import SimpleModel
from AI.Models.SimpleModel.SimpleModelTranslator import SimpleModelTranslator
from AI.Models.SimpleModel.SimpleModelEnvironment import SimpleModelEnvironment
from AI.Training.PPO import PPO
from AI.Training.Memory import Memory
from training_plotter import TrainingPlotter
import Constants
from Core.Domain.Entities.Map import Map


# ============================================================
# Модуль-уровневая функция воркера (обязательно для spawn)
# ============================================================

def _worker_loop(worker_id, cmd_queue, result_queue, target_steps, max_episode_steps, seed=None):
    """
    Постоянный воркер-процесс. Запускается один раз, ждёт команд через cmd_queue.
    Команды: (sd1, sd2) — собрать опыт; None — остановиться.

    Логика сбора:
      - Запускает полные эпизоды, которые заканчиваются только на забитом голе.
      - Продолжает собирать эпизоды пока step < target_steps * 2.
      - Останавливается на первом голе ПОСЛЕ того, как собрано >= target_steps шагов.
      - Если эпизод превысил max_episode_steps без гола — safety-truncation,
        эпизод сбрасывается, но сбор продолжается.
    """
    try:
        # Ограничиваем intra-op потоки PyTorch до 1 на воркер.
        # set_num_interop_threads НЕ вызываем: если пул уже инициализирован при импорте,
        # повторный вызов вызывает deadlock (документированное поведение PyTorch).
        torch.set_num_threads(1)

        if seed is not None:
            torch.manual_seed(seed + worker_id * 1000)
            np.random.seed(seed + worker_id * 1000)

        # Одноразовая инициализация — среда и модели создаются один раз
        map_obj = Map()
        map_obj.load_random()
        translator1 = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        translator2 = SimpleModelTranslator(map_obj, map_obj.players_team2[0], is_team_1=False)
        policy1 = SimpleModel(translator1)
        policy2 = SimpleModel(translator2)
        # num_steps = safety limit per episode (не лимит батча)
        env = SimpleModelEnvironment(policy1, policy2, num_steps=max_episode_steps)

        # Предаллоцируем numpy-буферы один раз на всё время жизни воркера.
        # Размер: target_steps*2 (минимальный батч) + max_episode_steps (один запасной эпизод).
        # Буферы переиспользуются каждый раунд — никаких выделений памяти в горячем цикле.
        S = Constants.state_size   # 19
        A = Constants.action_size  # 5
        max_buf = target_steps * 2 + max_episode_steps + 16
        buf_s1  = np.empty((max_buf, S), dtype=np.float32)
        buf_a1  = np.empty((max_buf, A), dtype=np.float32)
        buf_lp1 = np.empty(max_buf,     dtype=np.float32)
        buf_r1  = np.empty(max_buf,     dtype=np.float32)
        buf_t1  = np.empty(max_buf,     dtype=np.float32)
        buf_tr1 = np.empty(max_buf,     dtype=np.float32)
        buf_s2  = np.empty((max_buf, S), dtype=np.float32)
        buf_a2  = np.empty((max_buf, A), dtype=np.float32)
        buf_lp2 = np.empty(max_buf,     dtype=np.float32)
        buf_r2  = np.empty(max_buf,     dtype=np.float32)
        buf_t2  = np.empty(max_buf,     dtype=np.float32)
        buf_tr2 = np.empty(max_buf,     dtype=np.float32)

        while True:
            cmd = cmd_queue.get()
            if cmd is None:  # сигнал остановки
                break

            sd1, sd2 = cmd
            policy1.load_state_dict(sd1)
            policy2.load_state_dict(sd2)

            # Сброс среды
            s1, s2 = env.reset()
            step = 0
            n1 = 0  # счётчик записей P1
            n2 = 0  # счётчик записей P2

            total_reward    = 0.0
            goals_team1     = 0
            goals_team2     = 0
            kicks_team1     = 0
            kicks_team2     = 0
            goal_reward_sum = 0.0
            move_reward_sum = 0.0

            while True:
                with torch.inference_mode():
                    action1, log_prob1 = policy1.select_action(s1, deterministic=False)
                    action2, log_prob2 = policy2.select_action(s2, deterministic=False)

                (ns1, ns2), (r1, r2), _, info = env.step(action1, action2)

                is_truncated = info.get('truncated', False)
                nat_done     = info.get('natural_done', False)  # только гол

                if info.get('goal_team1', False): goals_team1 += 1
                if info.get('goal_team2', False): goals_team2 += 1
                if info.get('kick_team1', False): kicks_team1 += 1
                if info.get('kick_team2', False): kicks_team2 += 1

                r1v = r1.item()
                r2v = r2.item()
                total_reward += r1v + r2v

                if info.get('goal_team1', False):
                    goal_reward_sum += r1v
                else:
                    move_reward_sum += r1v

                term = 1.0 if nat_done else 0.0
                trunc = 1.0 if is_truncated else 0.0

                # Защита от переполнения буфера (если голов нет долго — принудительная остановка)
                if n1 >= max_buf or n2 >= max_buf:
                    break

                # Индексная запись в прелоцированные буферы — без .tolist(), без list.append
                buf_s1[n1]  = s1.detach().flatten().numpy()
                buf_a1[n1]  = action1.detach().flatten().numpy()
                buf_lp1[n1] = log_prob1.item()
                buf_r1[n1]  = r1v
                buf_t1[n1]  = term
                buf_tr1[n1] = trunc
                n1 += 1

                buf_s2[n2]  = s2.detach().flatten().numpy()
                buf_a2[n2]  = action2.detach().flatten().numpy()
                buf_lp2[n2] = log_prob2.item()
                buf_r2[n2]  = r2v
                buf_t2[n2]  = term
                buf_tr2[n2] = trunc
                n2 += 1

                step += 2

                if nat_done:
                    # Гол — проверяем, достаточно ли собрано шагов
                    if step >= target_steps * 2:
                        break  # батч готов, последний эпизод завершён голом
                    s1, s2 = env.reset()  # не хватает шагов — продолжаем
                elif is_truncated:
                    # Safety truncation без гола — сбрасываем и продолжаем
                    s1, s2 = env.reset()
                else:
                    s1, s2 = ns1, ns2

            # Отправляем срезы буферов — Queue пикклит их, копирование происходит один раз
            result_queue.put((
                worker_id,
                buf_s1[:n1], buf_a1[:n1], buf_lp1[:n1], buf_r1[:n1], buf_t1[:n1], buf_tr1[:n1],
                buf_s2[:n2], buf_a2[:n2], buf_lp2[:n2], buf_r2[:n2], buf_t2[:n2], buf_tr2[:n2],
                total_reward, goals_team1, goals_team2, kicks_team1, kicks_team2,
                goal_reward_sum, move_reward_sum,
            ))

    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put((
            worker_id,
            [], [], [], [], [], [],
            [], [], [], [], [], [],
            -1000.0, 0, 0, 0, 0, 0.0, 0.0,
        ))


# ============================================================
# Пул постоянных воркеров
# ============================================================

class WorkerPool:
    """Пул постоянных воркер-процессов. Создаётся один раз на всё обучение."""

    def __init__(self, num_workers, target_steps, max_episode_steps, seed=None):
        self.num_workers = num_workers
        self.cmd_queues   = [mp.Queue() for _ in range(num_workers)]
        self.result_queue = mp.Queue()
        self.processes    = []

        for i in range(num_workers):
            p = mp.Process(
                target=_worker_loop,
                args=(i, self.cmd_queues[i], self.result_queue, target_steps, max_episode_steps, seed),
                daemon=True,
            )
            p.start()
            self.processes.append(p)
        print(f"WorkerPool: запущено {num_workers} воркеров (pid: {[p.pid for p in self.processes]})")

    def collect(self, sd1, sd2):
        """Отправить задачу всем воркерам и дождаться результатов."""
        for cmd_q in self.cmd_queues:
            cmd_q.put((sd1, sd2))

        results = [None] * self.num_workers
        for _ in range(self.num_workers):
            result = self.result_queue.get()
            results[result[0]] = result
        return results

    def stop(self):
        """Остановить все воркеры."""
        for cmd_q in self.cmd_queues:
            cmd_q.put(None)
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        print("WorkerPool: все воркеры остановлены")


# ============================================================
# Сбор опыта
# ============================================================

def collect_experience_parallel(pool: WorkerPool, policy1, policy2, opponent_sd=None):
    """
    Параллельный сбор опыта через постоянный пул воркеров.
    Данные возвращаются напрямую через Queue (без mmap).
    """
    sd1 = policy1.state_dict()
    sd2 = opponent_sd if opponent_sd is not None else policy2.state_dict()

    results = pool.collect(sd1, sd2)

    all_states_p1,  all_actions_p1,  all_log_probs_p1  = [], [], []
    all_rewards_p1, all_terminals_p1, all_truncated_p1  = [], [], []
    all_states_p2,  all_actions_p2,  all_log_probs_p2  = [], [], []
    all_rewards_p2, all_terminals_p2, all_truncated_p2  = [], [], []

    total_reward  = 0.0
    total_goals_1 = 0
    total_goals_2 = 0
    total_kicks_1 = 0
    total_kicks_2 = 0
    total_goal_rew = 0.0
    total_move_rew = 0.0

    for r in results:
        if r is None:
            continue
        (_, sp1, ap1, lp1, rp1, tp1, trp1,
            sp2, ap2, lp2, rp2, tp2, trp2,
            rew, g1, g2, k1, k2, gr, mr) = r

        all_states_p1.extend(sp1);   all_actions_p1.extend(ap1)
        all_log_probs_p1.extend(lp1); all_rewards_p1.extend(rp1)
        all_terminals_p1.extend(tp1); all_truncated_p1.extend(trp1)

        all_states_p2.extend(sp2);   all_actions_p2.extend(ap2)
        all_log_probs_p2.extend(lp2); all_rewards_p2.extend(rp2)
        all_terminals_p2.extend(tp2); all_truncated_p2.extend(trp2)

        total_reward  += rew
        total_goals_1 += g1;  total_goals_2 += g2
        total_kicks_1 += k1;  total_kicks_2 += k2
        total_goal_rew += gr; total_move_rew += mr

    return (all_states_p1, all_actions_p1, all_log_probs_p1, all_rewards_p1, all_terminals_p1, all_truncated_p1,
            all_states_p2, all_actions_p2, all_log_probs_p2, all_rewards_p2, all_terminals_p2, all_truncated_p2,
            total_reward, total_goals_1, total_goals_2, total_kicks_1, total_kicks_2,
            total_goal_rew, total_move_rew)


# ============================================================
# Вспомогательные функции
# ============================================================

def create_environment_and_policies():
    """Создание среды и политик"""
    map_obj = Map()
    map_obj.load_random()

    translator1 = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
    translator2 = SimpleModelTranslator(map_obj, map_obj.players_team2[0], is_team_1=False)

    policy1 = SimpleModel(translator1)
    policy2 = SimpleModel(translator2)

    return policy1, policy2, translator1


# ============================================================
# Основной цикл тренировки
# ============================================================

def train_with_workers(num_episodes=200, num_workers=8, max_steps=1024,
                       save_interval=10, save_path="simple_model_approach_ball.pth",
                       weights_path=None, start_episode=0, lr=None,
                       phase1_frac=0.20, opponent_sync_every=10, pool_models=None):
    """
    Основной цикл тренировки с воркерами

    Args:
        num_episodes:       Количество эпизодов для этого запуска
        num_workers:        Количество воркеров для сбора данных
        max_steps:          Шагов на воркера
        save_interval:      Интервал сохранения
        save_path:          Путь сохранения модели
        weights_path:       Чекпоинт для дообучения (None = с нуля)
        start_episode:      Смещение эпизода для корректного курикулума и логов
        pool_models:        Список путей к моделям пула оппонентов
    """
    total_episodes = start_episode + num_episodes  # полный горизонт для курикулума

    print(f"\n{'='*70}")
    print(f"Запуск тренировки SimpleModel с воркерами")
    print(f"  - Эпизодов: {num_episodes}  (эп. {start_episode}–{total_episodes-1})")
    print(f"  - Воркеров: {num_workers}")
    print(f"  - Шагов на воркера: {max_steps}")
    print(f"  - Device: {Constants.device}")
    if weights_path:
        print(f"  - Чекпоинт: {weights_path}")
    print(f"{'='*70}\n")

    # Создаём политики
    policy1, policy2, _ = create_environment_and_policies()
    policy     = policy1  # Основная модель
    policy_old = policy2  # Старая модель для сбора опыта

    # Сохраняем случайные веса ДО загрузки чекпоинта — это и есть frozen opponent
    frozen_sd = {k: v.clone() for k, v in policy_old.state_dict().items()}

    # Загружаем веса из чекпоинта
    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=Constants.device)
        policy.load_state_dict(state)
        policy_old.load_state_dict(state)
        print(f"  Веса загружены: {weights_path}\n")
    elif weights_path:
        print(f"  [WARN] файл чекпоинта не найден: {weights_path}\n")

    # Загружаем пул оппонентов
    opponent_pool = []
    if pool_models:
        for path in pool_models:
            path = path.strip()
            if os.path.exists(path):
                sd = torch.load(path, map_location=Constants.device)
                opponent_pool.append(sd)
                print(f"  Пул: добавлен {path}")
            else:
                print(f"  Пул: [WARN] файл не найден: {path}")
        print(f"  Пул оппонентов: {len(opponent_pool)} модель(и)\n")

    # Создаём PPO
    ppo = PPO(policy, policy_old, episode_offset=start_episode, episodes_total=num_episodes)
    if lr is not None:
        for pg in ppo.optimizer.param_groups:
            pg['lr'] = lr
        ppo.scheduler.base_lrs = [lr for _ in ppo.scheduler.base_lrs]

    # Минимальный размер батча на воркера; реальный будет >= этого (до первого гола)
    target_steps      = max_steps
    max_episode_steps = max_steps * 4  # safety-truncation внутри одного эпизода
    min_total_samples = num_workers * target_steps * 2  # нижняя оценка (на практике больше)

    # Курикулум противника
    PHASE1_END          = int(total_episodes * phase1_frac)
    OPPONENT_SYNC_EVERY = opponent_sync_every

    print(f"  - Мин. шагов на воркера: {target_steps}  (safety-limit эпизода: {max_episode_steps})")
    print(f"  - Мин. размер батча: {min_total_samples}")
    print(f"  - Фаза 1 (эп. 0–{PHASE1_END-1}): противник заморожен")
    print(f"  - Фаза 2 (эп. {PHASE1_END}–{total_episodes-1}): синхронизация каждые {OPPONENT_SYNC_EVERY} эп.")
    print(f"{'='*70}\n")

    # Снепшот оппонента для lagged-фазы.
    # PPO внутри синхронизирует policy_old каждый эпизод, поэтому нельзя использовать
    # policy_old напрямую как "лагированного" оппонента — он всегда актуален.
    # Держим отдельную копию весов, которую обновляем только раз в OPPONENT_SYNC_EVERY эпизодов.
    lagged_sd = {k: v.clone() for k, v in policy.state_dict().items()}

    best_avg_reward = -float('inf')
    goals_history   = []
    plotter = TrainingPlotter(save_dir=os.path.dirname(save_path) or ".",
                              update_every=5)

    # Запускаем пул воркеров один раз на всё обучение
    pool = WorkerPool(num_workers, target_steps, max_episode_steps, seed=42)

    try:
        for episode in range(num_episodes):
            abs_ep       = start_episode + episode
            episode_start = time.time()

            # Определяем противника по фазе
            if abs_ep < PHASE1_END:
                opp_sd      = frozen_sd
                phase_label = "frozen"
                train_p2    = False
            elif opponent_pool and random.random() < 0.5:
                opp_sd      = random.choice(opponent_pool)
                phase_label = "pool"
                train_p2    = False
            else:
                opp_sd      = lagged_sd  # внешний снепшот, обновляется раз в N эп.
                phase_label = "lagged"
                train_p2    = False  # P2 собран lagged-политикой → IS-ratio некорректен для PPO

            # Собираем опыт через постоянный пул
            (states_p1, actions_p1, log_probs_p1, rewards_p1, terminals_p1, truncated_p1,
             states_p2, actions_p2, log_probs_p2, rewards_p2, terminals_p2, truncated_p2,
             total_reward, goals_1, goals_2, kicks_1, kicks_2,
             goal_rew, move_rew) = collect_experience_parallel(
                pool, policy, policy_old, opponent_sd=opp_sd)

            if len(states_p1) == 0 and len(states_p2) == 0:
                print(f"Episode {abs_ep}: Нет опыта для обучения!")
                continue

            # === ОБУЧЕНИЕ ===
            # P1 и P2 объединяются в один батч — актор и критик обновляются на всех данных.
            # В фазе frozen/pool P2 не обучается (данные только от P1).
            all_states   = states_p1   + (states_p2   if train_p2 else [])
            all_actions  = actions_p1  + (actions_p2  if train_p2 else [])
            all_logprobs = log_probs_p1 + (log_probs_p2 if train_p2 else [])
            all_rewards  = rewards_p1  + (rewards_p2  if train_p2 else [])
            all_terms    = terminals_p1 + (terminals_p2 if train_p2 else [])
            all_truncs   = truncated_p1 + (truncated_p2 if train_p2 else [])

            if len(all_states) > 0:
                memory = Memory()
                memory.states        = all_states
                memory.actions_final = all_actions
                memory.old_log_probs = all_logprobs
                memory.rewards       = all_rewards
                memory.is_terminals  = all_terms
                memory.is_truncated  = all_truncs
                memory.copy_to_tensors()
                label = "P1+P2" if train_p2 else "P1"
                ppo.update_combined(memory, abs_ep, minibatch_size=len(all_states) // 4,
                                    label=label, update_actor=True, step_scheduler=True)

            # Обновляем lagged-снепшот раз в N эпизодов в фазе 2.
            # policy_old трогать не нужно — PPO синхронизирует его сам каждый эпизод.
            if abs_ep >= PHASE1_END and abs_ep % OPPONENT_SYNC_EVERY == 0:
                lagged_sd = {k: v.clone() for k, v in policy.state_dict().items()}
                print(f"  [lagged opponent updated at ep {abs_ep}]")
                ppo.bump_entropy_on_lag_update(abs_ep)

            # Метрики
            all_rewards = rewards_p1 + rewards_p2
            avg_reward  = np.mean(all_rewards) if all_rewards else 0
            episode_time = time.time() - episode_start

            goals_history.append(goals_1 + goals_2)
            if len(goals_history) > 20:
                goals_history.pop(0)
            avg_goals = np.mean(goals_history)

            plotter.update(abs_ep + 1, goals_1, goals_2, kicks_1, kicks_2,
                           avg_reward, total_reward, len(states_p1) + len(states_p2))

            if episode % 5 == 0 or episode == num_episodes - 1:
                balance       = f"{goals_1/(goals_2+1e-3):.2f}" if goals_2 > 0 else "inf"
                total_rew_diag = goal_rew + move_rew if (goal_rew + move_rew) != 0 else 1.0
                goal_pct      = 100.0 * goal_rew / total_rew_diag if total_rew_diag > 0 else 0.0
                print(f"Episode {abs_ep + 1}/{total_episodes} [{phase_label}] | "
                      f"Reward: {avg_reward:.1f} | "
                      f"Goals: {goals_1}-{goals_2} (avg20: {avg_goals:.1f}, bal: {balance}) | "
                      f"Kicks: {kicks_1+kicks_2} | "
                      f"Rew breakdown: goal={goal_rew:.0f} ({goal_pct:.1f}%) move={move_rew:.0f} | "
                      f"Time: {episode_time:.2f}s")

            if abs_ep % save_interval == 0 and abs_ep > 0:
                torch.save(policy.state_dict(), save_path)
                print(f"  -> Модель сохранена: {save_path}")

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                dir_name  = os.path.dirname(save_path)
                base_name = os.path.basename(save_path)
                best_path = os.path.join(dir_name, "best_" + base_name) if dir_name else "best_" + save_path
                torch.save(policy.state_dict(), best_path)
                print(f"  -> BEST MODEL SAVED! Avg Reward: {best_avg_reward:.4f}")

    finally:
        pool.stop()

    torch.save(policy.state_dict(), save_path)
    plotter.close()
    print(f"\n{'='*70}")
    print(f"Тренировка завершена!")
    print(f"  - Лучшая средняя награда: {best_avg_reward:.4f}")
    print(f"  - Модель сохранена: {save_path}")
    print(f"{'='*70}\n")

    return policy


# ============================================================
# Точка входа
# ============================================================

def main():
    import argparse
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Train SimpleModel")
    parser.add_argument("--weights",        default=None,
                        help="Чекпоинт для дообучения (*.pth)")
    parser.add_argument("--start-episode",  type=int, default=0)
    parser.add_argument("--episodes",       type=int, default=500)
    parser.add_argument("--workers",        type=int, default=6)
    parser.add_argument("--steps",          type=int, default=1024)
    parser.add_argument("--save",           default="models/simple_model_vs.pth")
    parser.add_argument("--save-interval",  type=int, default=10)
    parser.add_argument("--lr",             type=float, default=None)
    parser.add_argument("--phase1-frac",    type=float, default=0.20)
    parser.add_argument("--opponent-sync",  type=int, default=10)
    parser.add_argument("--pool-models",    default=None,
                        help="Пути к моделям пула через запятую (напр. models/vs_1.pth,models/vs_3.pth)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)

    # Логирование: дублируем stdout в файл рядом с моделью
    log_dir  = os.path.dirname(args.save) or "."
    log_name = os.path.splitext(os.path.basename(args.save))[0] + ".log"
    tee = _Tee(os.path.join(log_dir, log_name))

    pool_list = [p.strip() for p in args.pool_models.split(",")] if args.pool_models else None

    try:
        train_with_workers(
            num_episodes        = args.episodes,
            num_workers         = args.workers,
            max_steps           = args.steps,
            save_interval       = args.save_interval,
            save_path           = args.save,
            weights_path        = args.weights,
            start_episode       = args.start_episode,
            lr                  = args.lr,
            phase1_frac         = args.phase1_frac,
            opponent_sync_every = args.opponent_sync,
            pool_models         = pool_list,
        )
        print("Тренировка завершена! Модель готова к использованию.")
    finally:
        tee.close()


if __name__ == "__main__":
    main()
