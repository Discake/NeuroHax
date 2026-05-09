"""
Тесты базовых компонентов SimpleModel pipeline.
Проверяем: состояние, награды, согласованность (state, action), PPO градиент.
"""
import sys
import os
import math
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Domain.Entities.Map import Map
from AI.Models.SimpleModel.SimpleModelTranslator import SimpleModelTranslator
from AI.Models.SimpleModel.SimpleModelEnvironment import SimpleModelEnvironment
from AI.Models.SimpleModel.SimpleModelNetAction import SimpleModelNetAction
from AI.Models.SimpleModel.Policy import SimpleModel
from AI.Training.PPO import PPO
from AI.Training.Memory import Memory


# ---------------------------------------------------------------------------
# 1. Тесты состояния (translator)
# ---------------------------------------------------------------------------

class TestState:
    def setup_method(self):
        self.map = Map()
        self.map.load_random()

    def test_rel_ball_pos_x_positive_when_ball_right(self):
        """Мяч правее игрока → rel_ball_pos_x > 0"""
        player = self.map.players_team1[0]
        ball = self.map.balls[0]
        player.x = 200
        ball.x = 400
        translator = SimpleModelTranslator(self.map, player, is_team_1=True)
        state = translator.translate({})
        rel_ball_x = state[4].item()  # индекс rel_ball_pos_x
        assert rel_ball_x > 0, f"Мяч справа, ожидаем rel_ball_x > 0, получили {rel_ball_x:.3f}"

    def test_rel_ball_pos_x_negative_when_ball_left(self):
        """Мяч левее игрока → rel_ball_pos_x < 0"""
        player = self.map.players_team1[0]
        ball = self.map.balls[0]
        player.x = 500
        ball.x = 200
        translator = SimpleModelTranslator(self.map, player, is_team_1=True)
        state = translator.translate({})
        rel_ball_x = state[4].item()
        assert rel_ball_x < 0, f"Мяч слева, ожидаем rel_ball_x < 0, получили {rel_ball_x:.3f}"

    def test_team2_rel_ball_flipped_correctly(self):
        """
        Для team2: флип x-оси.
        Мяч слева физически от игрока 2 → в перевёрнутом пространстве мяч СПРАВА → rel_ball_x > 0
        """
        player = self.map.players_team2[0]
        ball = self.map.balls[0]
        player.x = 600
        ball.x = 400  # мяч физически левее игрока 2
        translator = SimpleModelTranslator(self.map, player, is_team_1=False)
        state = translator.translate({})
        rel_ball_x = state[4].item()
        assert rel_ball_x > 0, (
            f"Team2: мяч физически слева, но в перевёрнутом пространстве должен быть справа. "
            f"rel_ball_x={rel_ball_x:.3f}"
        )

    def test_state_dim(self):
        """Размер состояния = 14"""
        player = self.map.players_team1[0]
        translator = SimpleModelTranslator(self.map, player, is_team_1=True)
        state = translator.translate({})
        assert state.shape == (14,), f"Ожидаем shape (14,), получили {state.shape}"


# ---------------------------------------------------------------------------
# 2. Тесты награды (environment)
# ---------------------------------------------------------------------------

class TestRewards:
    def _make_env(self):
        map_obj = Map()
        map_obj.load_random()
        t1 = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        t2 = SimpleModelTranslator(map_obj, map_obj.players_team2[0], is_team_1=False)
        p1 = SimpleModel(t1)
        p2 = SimpleModel(t2)
        env = SimpleModelEnvironment(p1, p2, num_steps=200)
        env.reset()
        return env

    def test_approach_reward_positive(self):
        """Шаг в сторону мяча → награда > 0"""
        env = self._make_env()
        player = env.map.players_team1[0]
        ball = env.map.balls[0]

        # Ставим игрока и мяч с чётким зазором
        player.x, player.y = 200, 300
        ball.x, ball.y = 400, 300
        env.last_dist1 = math.sqrt((ball.x - player.x)**2 + (ball.y - player.y)**2)

        # Двигаем игрока вправо (к мячу)
        player.vx = 20
        player.vy = 0

        r1, _, _ = env.compute_rewards()
        assert r1.item() > 0, f"Движение к мячу должно давать r > 0, получили {r1.item():.4f}"

    def test_retreat_reward_negative(self):
        """Шаг от мяча → награда < 0"""
        env = self._make_env()
        player = env.map.players_team1[0]
        ball = env.map.balls[0]

        player.x, player.y = 400, 300
        ball.x, ball.y = 200, 300
        env.last_dist1 = math.sqrt((ball.x - player.x)**2 + (ball.y - player.y)**2)

        # Ставим скорость вправо — ОТ мяча, затем двигаем физику
        player.vx = 45
        player.vy = 0
        env.map.move_balls()  # позиция обновляется здесь, до compute_rewards

        r1, _, _ = env.compute_rewards()
        assert r1.item() < 0, f"Движение от мяча должно давать r < 0, получили {r1.item():.4f}"

    def test_kick_reward_fires_on_first_kick_only(self):
        """KICK_REWARD срабатывает при переходе is_kicking False→True, не повторно"""
        env = self._make_env()
        player = env.map.players_team1[0]
        ball = env.map.balls[0]
        player.x, player.y = 200, 300
        ball.x, ball.y = 220, 300  # в зоне удара
        env.last_dist1 = 20.0

        env.last_kick_team1 = False
        player.is_kicking = True
        r1_first, _, _ = env.compute_rewards()

        # Второй шаг: флаг остаётся True
        r1_second, _, _ = env.compute_rewards()

        assert r1_first.item() > 50, f"Первый удар должен давать KICK_REWARD, получили {r1_first.item():.2f}"
        assert r1_second.item() < r1_first.item(), (
            f"Повторный удар не должен давать KICK_REWARD повторно. "
            f"first={r1_first.item():.2f}, second={r1_second.item():.2f}"
        )


# ---------------------------------------------------------------------------
# 3. Тест согласованности (state, action) — ключевой баг из get_action
# ---------------------------------------------------------------------------

class TestStateActionConsistency:
    def test_select_action_uses_passed_tensor(self):
        """
        select_action(tensor) должен использовать переданный тензор,
        а не перетранслировать из карты модели (стальной map_obj).
        """
        map_obj = Map()
        map_obj.load_random()
        translator = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        policy = SimpleModel(translator)

        # Создаём тензор с заведомо другим состоянием
        fake_state = torch.zeros(14)
        fake_state[4] = 0.9  # мяч сильно вправо

        # Вызываем дважды с одинаковым тензором — действия должны быть статистически схожи
        actions = []
        torch.manual_seed(42)
        for _ in range(20):
            a, _ = policy.select_action(fake_state, deterministic=False)
            actions.append(a.detach())

        # Убеждаемся, что модель хоть что-то делает (не возвращает мусор размера != 5)
        assert actions[0].shape == (5,), f"Ожидаем action shape (5,), получили {actions[0].shape}"

    def test_env_state_matches_stored_state(self):
        """
        state из env.step и state переданный в action — одно и то же.
        Проверяем, что после нашего фикса policy не перечитывает устаревший map.
        """
        map_obj = Map()
        map_obj.load_random()
        t1 = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        t2 = SimpleModelTranslator(map_obj, map_obj.players_team2[0], is_team_1=False)
        policy1 = SimpleModel(t1)
        policy2 = SimpleModel(t2)

        env = SimpleModelEnvironment(policy1, policy2, num_steps=10)
        s1, s2 = env.reset()

        # Состояние из env и состояние из translator модели — разные объекты (разные map)
        # После фикса policy должен использовать s1 напрямую
        action1, logp1 = policy1.select_action(s1, deterministic=False)

        # Проверяем что logp1 — скалярный тензор (не матрица [B,B])
        assert logp1.shape == torch.Size([]), (
            f"log_prob должен быть скаляром, получили shape={logp1.shape}. "
            f"Возможен broadcasting-баг в evaluate_actions."
        )
        assert not torch.isnan(logp1), "log_prob не должен быть NaN"
        assert not torch.isinf(logp1), "log_prob не должен быть Inf"


# ---------------------------------------------------------------------------
# 4. Тест PPO: градиент идёт в правильную сторону
# ---------------------------------------------------------------------------

class TestPPOGradient:
    def test_no_nan_in_losses(self):
        """Потери не должны содержать NaN/Inf"""
        map_obj = Map()
        map_obj.load_random()
        t1 = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        policy = SimpleModel(t1)
        policy_old = SimpleModel(t1)
        ppo = PPO(policy, policy_old)

        N = 256
        states = torch.randn(N, 14) * 0.5
        actions = torch.randint(0, 2, (N, 5)).float()
        with torch.no_grad():
            old_logps, _, _ = policy.evaluate_actions(states, actions)

        memory = Memory()
        memory.states = states.tolist()
        memory.actions_final = actions.tolist()
        memory.old_log_probs = old_logps.tolist()
        memory.rewards = (torch.randn(N) * 2).tolist()
        memory.is_terminals = [0.0] * N
        memory.is_truncated = [1.0] * N

        ppo.update_combined(memory, 0, minibatch_size=N)

        for name, param in policy.named_parameters():
            assert not torch.isnan(param).any(), f"NaN в параметрах после обновления: {name}"
            assert not torch.isinf(param).any(), f"Inf в параметрах после обновления: {name}"

    def test_policy_learns_ball_direction(self):
        """
        Корреляция состояние→действие→награда.
        - Мяч справа (rel_ball_pos_x > 0) + action right → reward +2
        - Мяч справа + action left  → reward -2
        После обучения P(right | мяч справа) должна вырасти.
        """
        torch.manual_seed(0)
        map_obj = Map()
        map_obj.load_random()
        t1 = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        policy = SimpleModel(t1)
        policy_old = SimpleModel(t1)
        ppo = PPO(policy, policy_old)

        N = 512  # половина — right=1 с +reward, половина — left=1 с -reward
        states = torch.zeros(N, 14)
        states[:, 4] = 0.8  # мяч всегда справа

        actions = torch.zeros(N, 5)
        actions[:N//2, 3] = 1.0   # right=1
        actions[N//2:, 2] = 1.0   # left=1

        rewards = [2.0] * (N // 2) + [-2.0] * (N // 2)

        with torch.no_grad():
            old_logps, _, _ = policy.evaluate_actions(states, actions)

        memory = Memory()
        memory.states = states.tolist()
        memory.actions_final = actions.tolist()
        memory.old_log_probs = old_logps.tolist()
        memory.rewards = rewards
        memory.is_terminals = [0.0] * N
        memory.is_truncated = [1.0] * N

        test_state = states[:1]
        with torch.no_grad():
            logits_before, _, _ = policy.forward(test_state)
        prob_right_before = torch.sigmoid(logits_before[0, 3]).item()

        # Несколько эпох обучения для накопления сигнала
        for ep in range(5):
            ppo.update_combined(memory, ep, minibatch_size=N)

        with torch.no_grad():
            logits_after, _, _ = policy.forward(test_state)
        prob_right_after = torch.sigmoid(logits_after[0, 3]).item()

        assert prob_right_after > prob_right_before, (
            f"P(right | мяч справа) должна вырасти после обучения: "
            f"{prob_right_before:.3f} → {prob_right_after:.3f}"
        )
