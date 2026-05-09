"""
Тесты для Environment.py - расчёт потенциалов и reward shaping
"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AI.Training.Environment import Environment
from AI.SeparateNetworkPolicy import SeparateNetworkPolicy


class TestEnvironmentPotentials:
    """Тесты для расчёта потенциалов"""

    @pytest.fixture
    def environment(self):
        """Создаёт тестовое окружение"""
        nn1 = SeparateNetworkPolicy()
        nn2 = SeparateNetworkPolicy()
        return Environment(nn1, nn2, num_steps=100)

    def test_compute_potentials_basic(self, environment):
        """Базовый тест расчёта потенциалов"""
        player = type('Player', (), {
            'position': torch.tensor([100.0, 100.0]),
            'velocity': torch.tensor([0.1, 0.2])
        })()
        
        ball = type('Ball', (), {
            'position': torch.tensor([200.0, 150.0]),
            'velocity': torch.tensor([0.5, 0.0])
        })()
        
        goal = torch.tensor([400.0, 150.0])
        
        potential = environment._compute_potentials(player, ball, goal)
        
        assert torch.is_tensor(potential) or isinstance(potential, float)
        assert potential < 0  # Потенциал должен быть отрицательным (расстояние до ворот)

    def test_compute_potentials_zero_distance(self, environment):
        """Потенциал при нулевой дистанции до ворот"""
        player = type('Player', (), {
            'position': torch.tensor([100.0, 100.0]),
            'velocity': torch.tensor([0.0, 0.0])
        })()
        
        ball = type('Ball', (), {
            'position': torch.tensor([400.0, 150.0]),  # На линии ворот
            'velocity': torch.tensor([0.0, 0.0])
        })()
        
        goal = torch.tensor([400.0, 150.0])  # Мяч на воротах
        
        potential = environment._compute_potentials(player, ball, goal)
        
        # Потенциал должен быть максимальным (близким к 0)
        assert potential <= 0

    def test_compute_potentials_boundary_penalty(self, environment):
        """Штраф за выход за границы"""
        player = type('Player', (), {
            'position': torch.tensor([-50.0, 100.0]),  # За границей
            'velocity': torch.tensor([0.0, 0.0])
        })()
        
        ball = type('Ball', (), {
            'position': torch.tensor([100.0, 100.0]),
            'velocity': torch.tensor([0.0, 0.0])
        })()
        
        goal = torch.tensor([400.0, 150.0])
        
        potential = environment._compute_potentials(player, ball, goal)
        
        # Должен быть большой штраф
        assert potential < -2.0

    def test_compute_potentials_activity_bonus(self, environment):
        """Бонус за активность игрока"""
        player_slow = type('Player', (), {
            'position': torch.tensor([100.0, 100.0]),
            'velocity': torch.tensor([0.0, 0.0])  # Стоит
        })()
        
        player_fast = type('Player', (), {
            'position': torch.tensor([100.0, 100.0]),
            'velocity': torch.tensor([0.4, 0.0])  # Движется
        })()
        
        ball = type('Ball', (), {
            'position': torch.tensor([200.0, 150.0]),
            'velocity': torch.tensor([0.0, 0.0])
        })()
        
        goal = torch.tensor([400.0, 150.0])
        
        pot_slow = environment._compute_potentials(player_slow, ball, goal)
        pot_fast = environment._compute_potentials(player_fast, ball, goal)
        
        # Активный игрок должен иметь больший потенциал
        assert pot_fast > pot_slow

    def test_compute_potentials_ball_velocity_to_goal(self, environment):
        """Влияние скорости мяча в сторону ворот"""
        player = type('Player', (), {
            'position': torch.tensor([100.0, 100.0]),
            'velocity': torch.tensor([0.0, 0.0])
        })()
        
        ball_towards = type('Ball', (), {
            'position': torch.tensor([200.0, 150.0]),
            'velocity': torch.tensor([0.5, 0.0])  # К воротам
        })()
        
        ball_away = type('Ball', (), {
            'position': torch.tensor([200.0, 150.0]),
            'velocity': torch.tensor([-0.5, 0.0])  # От ворот
        })()
        
        goal = torch.tensor([400.0, 150.0])
        
        pot_towards = environment._compute_potentials(player, ball_towards, goal)
        pot_away = environment._compute_potentials(player, ball_away, goal)
        
        # Мяч, летящий к воротам, должен давать больший потенциал
        assert pot_towards > pot_away


class TestEnvironmentRewards:
    """Тесты для системы наград"""

    @pytest.fixture
    def environment(self):
        """Окружение с инициализацией"""
        nn1 = SeparateNetworkPolicy()
        nn2 = SeparateNetworkPolicy()
        env = Environment(nn1, nn2, num_steps=100)
        env.reset()
        return env

    def test_improved_rewards_goal_scored_team1(self, environment):
        """Награда за гол команды 1"""
        environment.map.score_team1 = True
        
        r1, r2, done = environment.improved_rewards()
        
        assert done == True
        assert r1.item() >= 70.0  # Награда за гол
        assert r2.item() <= -70.0  # Штраф за пропущенный гол

    def test_improved_rewards_goal_scored_team2(self, environment):
        """Награда за гол команды 2"""
        environment.map.score_team2 = True
        
        r1, r2, done = environment.improved_rewards()
        
        assert done == True
        assert r2.item() >= 70.0
        assert r1.item() <= -70.0

    def test_reset_initializes_potentials(self, environment):
        """Сброс должен инициализировать потенциалы"""
        environment.reset()
        
        assert hasattr(environment, 'last_potential_team1')
        assert hasattr(environment, 'last_potential_team2')

    def test_step_returns_correct_structure(self, environment):
        """Шаг должен возвращать правильную структуру"""
        environment.reset()
        
        action = torch.tensor([0.1, 0.2, 0.0])
        result = environment.step(action, action)
        
        states, rewards, done, info = result
        
        assert len(states) == 2  # (s1, s2)
        assert len(rewards) == 2  # (r1, r2)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert 'truncated' in info
        assert 'natural_done' in info


class TestEnvironmentTruncated:
    """Тесты для truncated логики"""

    @pytest.fixture
    def environment(self):
        """Окружение с коротким лимитом шагов"""
        nn1 = SeparateNetworkPolicy()
        nn2 = SeparateNetworkPolicy()
        return Environment(nn1, nn2, num_steps=10)

    def test_truncated_after_max_steps(self, environment):
        """Truncated должен срабатывать после max_steps"""
        environment.reset()
        
        action = torch.tensor([0.1, 0.2, 0.0])
        
        # Делаем 9 шагов
        for _ in range(9):
            _, _, done, info = environment.step(action, action)
            if done:
                break
        
        # 10-й шаг должен вызвать truncated
        _, _, done, info = environment.step(action, action)
        
        assert info.get('truncated') == True

    def test_natural_done_priority(self, environment):
        """Natural done имеет приоритет над truncated"""
        environment.reset()
        environment.map.score_team1 = True
        
        action = torch.tensor([0.1, 0.2, 0.0])
        _, _, done, info = environment.step(action, action)
        
        assert info.get('natural_done') == True
        assert info.get('truncated') == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
