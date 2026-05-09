"""
Запуск визуализации обученной модели NeuroHax

Использование:
    python run_visualization.py --model models/test_model.pt
    python run_visualization.py --model models/test_model.pt --speed 2
    python run_visualization.py --model models/test_model.pt --games 10
"""

import argparse
import sys
import os
import time
import random

import torch

# Добавляем корень проекта в path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from AI.Models.SimpleModel.Policy import SimpleModel
from AI.Models.SimpleModel.SimpleModelTranslator import SimpleModelTranslator
from AI.Models.SimpleModel.SimpleModelEnvironment import create_simple_model_environment, SimpleModelEnvironment
import Constants
from AI.SeparateNetworkPolicy import SeparateNetworkPolicy
from AI.Training.Environment import Environment
from visualize_game import GameVisualizer
from Core.Domain.Entities.Map import Map
from Player_actions.HumanPlayerController import HumanPlayerController


class Colors:
    """ANSI цвета для консоли"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_info(text):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def print_success(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def load_model(model_path, map_obj=None, team_id=0, player_id=0):
    """Загрузка модели SimpleModel"""
    if not os.path.exists(model_path):
        print_error(f"Модель не найдена: {model_path}")
        return None

    print_info(f"Загрузка модели: {model_path}")

    # Для SimpleModel нужен translator
    if map_obj is None:
        map_obj = Map()
        map_obj.load_random()
    
    is_team_1 = (team_id == 0)
    translator = SimpleModelTranslator(map_obj, map_obj.players_team1[0] if is_team_1 else map_obj.players_team2[0], is_team_1)
    
    policy = SimpleModel(translator)

    try:
        state_dict = torch.load(model_path, map_location=Constants.device, weights_only=True)
        policy.load_state_dict(state_dict)
        print_success("Модель загружена")
        return policy
    except Exception as e:
        print_error(f"Ошибка загрузки модели: {e}")
        return None


def run_visualization(args):
    """Запуск визуализации для SimpleModel"""
    human = getattr(args, 'human', None)  # None / 1 / 2
    human_ctrl = None

    if human == 1:
        print_header("🎮 NeuroHax — Player vs AI (you are RED / team 1)")
        human_ctrl = HumanPlayerController(is_team_1=True)
    elif human == 2:
        print_header("🎮 NeuroHax — AI vs Player (you are BLUE / team 2)")
        human_ctrl = HumanPlayerController(is_team_1=False)
    else:
        print_header("🎮 NeuroHax Visualization (SimpleModel)")

    # Создаём карту для инициализации
    map_obj = Map()
    map_obj.load_random()

    # Загрузка модели (обязательна, если не оба — живые)
    if args.model:
        policy1 = load_model(args.model, map_obj, team_id=0)
        if policy1 is None:
            return 1
        # Создаём вторую политику
        if args.opponent_model and os.path.exists(args.opponent_model):
            policy2 = load_model(args.opponent_model, map_obj, team_id=1)
        else:
            print_info("Используется та же модель для противника (симметричная игра)")
            policy2 = SimpleModel(
                SimpleModelTranslator(map_obj, map_obj.players_team2[0], is_team_1=False)
            )
            policy2.load_state_dict(policy1.state_dict())
    else:
        # Режим без модели: обе политики случайные (для human vs human или отладки)
        translator1 = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        translator2 = SimpleModelTranslator(map_obj, map_obj.players_team2[0], is_team_1=False)
        policy1 = SimpleModel(translator1)
        policy2 = SimpleModel(translator2)

    # Создание среды для SimpleModel
    print_info("Создание среды для SimpleModel...")
    env = SimpleModelEnvironment(policy1, policy2, num_steps=2048)

    # Создание визуализатора
    title = f"NeuroHax - {os.path.basename(args.model) if args.model else 'Demo'}"
    visualizer = GameVisualizer(
        width=Constants.window_size[0],
        height=Constants.window_size[1],
        title=title
    )
    if human == 1:
        visualizer.human_controls_hint = 'team1'
    elif human == 2:
        visualizer.human_controls_hint = 'team2'

    # Тепловая карта потенциала (H для включения в игре)
    visualizer.set_potential_params(
        sigma_x=env._sigma_x,
        sigma_y=env._sigma_y,
        goal_cy=env._goal_cy,
        field_w=env._field_width,
        field_h=env._field_height,
    )

    # Ожидание старта
    if not visualizer.wait_for_start():
        visualizer.close()
        return 0

    # Статистика
    wins = 0
    losses = 0
    draws = 0
    total_reward = 0.0
    episode = 0

    if human:
        print_info("Управление: P - пауза, I - инфо, V - векторы, R - сброс, ESC - выход")
    else:
        print_info("Управление: SPACE - пауза, I - инфо, V - векторы, R - сброс, ESC - выход")

    # В human-режиме скорость зафиксирована на 60 FPS
    fps_cap = 60 if human else args.speed * 60

    running = True
    while running and episode < args.games:
        # Сброс среды
        state1, state2 = env.reset()
        episode_reward = 0.0
        step = 0
        done = False

        while running and not done:
            # Обработка событий
            event_result = visualizer.handle_events()

            if event_result == 'reset':
                break  # Сброс эпизода

            if event_result == False:
                running = False
                break

            if visualizer.paused:
                visualizer.clock.tick(60)
                continue

            # Выбор действий
            if human == 1:
                action1 = human_ctrl.get_action()
                with torch.no_grad():
                    action2, _ = policy2.select_action(state2, deterministic=False)
            elif human == 2:
                with torch.no_grad():
                    action1, _ = policy1.select_action(state1, deterministic=False)
                action2 = human_ctrl.get_action()
            else:
                with torch.no_grad():
                    action1, _ = policy1.select_action(state1, deterministic=False)
                    action2, _ = policy2.select_action(state2, deterministic=False)

            # Шаг в среде
            (next_state1, next_state2), (reward1, reward2), done, info = env.step(action1, action2)

            state1, state2 = next_state1, next_state2
            episode_reward += reward1.item()
            step += 1

            # Отрисовка
            score1 = env.map.score_team1
            score2 = env.map.score_team2

            visualizer.draw(
                env.map,
                score1, score2,
                episode=episode + 1,
                step=step,
                reward=episode_reward
            )

            # Контроль скорости
            visualizer.clock.tick(fps_cap)

        # Статистика эпизода
        natural_done = info.get('natural_done', False)

        if reward1.item() > reward2.item():
            wins += 1
            result = "🏆 Победа"
        elif reward1.item() < reward2.item():
            losses += 1
            result = "❌ Поражение"
        else:
            draws += 1
            result = "🤝 Ничья"

        total_reward += episode_reward
        episode += 1

        print(f"{Colors.OKGREEN}Эпизод {episode}: {result} | "
              f"Награда: {episode_reward:.2f} | "
              f"Шагов: {step} | "
              f"Счёт: {score1}:{score2}{Colors.ENDC}")

        # Пауза между эпизодами
        if episode < args.games and running:
            time.sleep(1.0)

    # Финальная статистика
    visualizer.close()

    print()
    print_header("Результаты")
    print(f"Всего эпизодов: {episode}")
    print(f"Победы: {wins} ({wins/episode*100:.1f}%)")
    print(f"Поражения: {losses} ({losses/episode*100:.1f}%)")
    print(f"Ничьи: {draws} ({draws/episode*100:.1f}%)")
    print(f"Средняя награда: {total_reward/episode:.4f}")

    return 0


def run_demo(args):
    """Демонстрация со случайными SimpleModel политиками"""
    print_header("🎮 NeuroHax Demo (SimpleModel)")

    # Создаём карту и среду для SimpleModel
    env, policy1, policy2 = create_simple_model_environment(num_steps=1024)

    # Создание визуализатора
    visualizer = GameVisualizer(
        width=Constants.window_size[0],
        height=Constants.window_size[1],
        title="NeuroHax Demo - SimpleModel"
    )

    # Ожидание старта
    if not visualizer.wait_for_start():
        visualizer.close()
        return 0

    print_info("Демонстрационный режим (случайные SimpleModel политики)")

    running = True
    episode = 0

    while running and episode < args.games:
        # Сброс среды
        state1, state2 = env.reset()
        step = 0
        done = False

        while running and not done:
            # Обработка событий
            event_result = visualizer.handle_events()

            if event_result == 'reset':
                break

            if event_result == False:
                running = False
                break

            if visualizer.paused:
                visualizer.clock.tick(60)
                continue

            # Случайные бинарные действия
            action1 = torch.randint(0, 2, (5,), device=Constants.device).float()
            action2 = torch.randint(0, 2, (5,), device=Constants.device).float()

            # Инверсия для симметрии
            action2_inv = action2.clone()
            action2_inv[0] = 1 - action2_inv[0]  # Инверсия up/down
            action2_inv[2] = 1 - action2_inv[2]  # Инверсия left/right

            # Шаг в среде
            (next_state1, next_state2), (reward1, reward2), done, info = env.step(action1, action2_inv)

            state1, state2 = next_state1, next_state2
            step += 1

            # Отрисовка
            visualizer.draw(
                env.map,
                env.map.score_team1,
                env.map.score_team2,
                episode=episode + 1,
                step=step,
                reward=0.0
            )

            visualizer.clock.tick(args.speed * 60)

        episode += 1

        if episode < args.games and running:
            time.sleep(0.5)

    visualizer.close()
    print_info(f"Показано {episode} эпизодов")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="NeuroHax - Визуализация обученной модели (SimpleModel)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s --model models/simple_model_approach_ball.pth
  %(prog)s --model models/simple_model_approach_ball.pth --speed 2
  %(prog)s --model models/simple_model_approach_ball.pth --games 10
  %(prog)s --demo
        """
    )

    # Режим работы
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--model", type=str, help="Путь к файлу модели SimpleModel (.pt)")
    mode_group.add_argument("--demo", action="store_true", help="Демонстрационный режим (SimpleModel)")

    # Параметры
    parser.add_argument("--opponent-model", type=str, help="Модель противника")
    parser.add_argument("--games", type=int, default=10, help="Количество игр")
    parser.add_argument("--max-steps", type=int, default=1024, help="Макс. шагов на игру")
    parser.add_argument("--speed", type=int, default=1, help="Скорость (1 = 60 FPS)")
    parser.add_argument("--human", type=int, choices=[1, 2], default=None,
                        help="Играть самому: 1 = красный (WASD+Space), 2 = синий (стрелки+Enter)")

    args = parser.parse_args()

    # Проверка аргументов
    if args.model or args.human:
        return run_visualization(args)
    elif args.demo:
        return run_demo(args)
    else:
        print_error("Укажите --model, --human или --demo")
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())
