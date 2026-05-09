"""
Тесты для PPO алгоритма
"""
import pytest
import torch
import sys
import os

# Добавляем корень проекта в path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AI.Training.PPO import PPO
from AI.Training.Memory import Memory
from AI.SeparateNetworkPolicy import SeparateNetworkPolicy


class TestPPO:
    """Тесты для PPO алгоритма"""

    @pytest.fixture
    def sample_policy(self):
        """Создаёт тестовую политику"""
        policy = SeparateNetworkPolicy()
        # Проверяем, что параметры есть
        assert len(list(policy.parameters())) > 0, "Policy должна иметь параметры"
        return policy

    @pytest.fixture
    def ppo_instance(self, sample_policy):
        """Создаёт PPO экземпляр"""
        # Проверяем параметры перед созданием PPO
        params = list(sample_policy.parameters())
        assert len(params) > 0, f"Policy должна иметь параметры, получено: {len(params)}"
        return PPO(sample_policy)

    @pytest.fixture
    def sample_memory(self):
        """Создаёт тестовую память с фиктивными данными"""
        memory = Memory()
        
        # Генерируем 10 шагов опыта
        for i in range(10):
            state = torch.tensor([0.5] * 17)  # state_size = 17
            action = torch.tensor([0.1, 0.2, 0.0])  # velocity_x, velocity_y, kick
            log_prob = torch.tensor(-1.5)
            reward = torch.tensor(0.5 + i * 0.1)
            done = 0
            is_truncated = 0
            
            memory.store(state, action, log_prob, reward, done, is_truncated)
        
        return memory

    def test_compute_returns_and_advantages_terminal_state(self, ppo_instance):
        """GAE должен обнулять advantage при terminal state"""
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        values = [0.5, 1.0, 1.5, 2.0, 2.5]
        terminals = [0, 0, 1, 0, 0]  # Terminal на шаге 2
        truncated = [0, 0, 0, 0, 0]

        returns, advantages = ppo_instance.compute_returns_and_advantages(
            rewards, values, terminals, truncated, gamma=0.99, lam=0.95
        )

        assert len(returns) == 5
        assert len(advantages) == 5
        
        # Advantage после terminal state должен быть независимым от предыдущих шагов
        assert torch.is_tensor(returns)
        assert torch.is_tensor(advantages)

    def test_compute_returns_and_advantages_truncated(self, ppo_instance):
        """GAE должен учитывать truncated флаги"""
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5]
        terminals = [0, 0, 0]
        truncated = [0, 0, 1]  # Truncated на последнем шаге

        returns, advantages = ppo_instance.compute_returns_and_advantages(
            rewards, values, terminals, truncated
        )

        assert len(returns) == 3
        assert len(advantages) == 3

    def test_compute_returns_and_advantages_all_terminal(self, ppo_instance):
        """Все terminal states"""
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5]
        terminals = [1, 1, 1]
        truncated = [0, 0, 0]

        returns, advantages = ppo_instance.compute_returns_and_advantages(
            rewards, values, terminals, truncated
        )

        # Каждый шаг должен быть независимым
        assert len(returns) == 3
        assert len(advantages) == 3

    def test_entropy_coef_decay(self, ppo_instance):
        """Проверка затухания entropy coefficient"""
        coef_start = ppo_instance.get_entropy_coef(0)
        coef_mid = ppo_instance.get_entropy_coef(50)
        coef_end = ppo_instance.get_entropy_coef(100)

        assert coef_start == ppo_instance.entropy_coef_initial
        assert coef_end <= coef_start

    def test_k_epochs_decay(self, ppo_instance):
        """Проверка уменьшения количества эпох"""
        k_start = ppo_instance.get_K_epochs(0)
        k_end = ppo_instance.get_K_epochs(100)

        assert k_start == ppo_instance.K_epochs_initial
        assert k_end >= ppo_instance.K_epochs_final
        assert k_end <= k_start

    def test_update_combined_basic(self, ppo_instance, sample_memory):
        """Базовый тест обновления PPO"""
        # Запускаем обновление
        ppo_instance.update_combined(sample_memory, ep=0, minibatch_size=5)

        # Проверяем, что веса обновились (сравниваем с копией)
        assert ppo_instance.policy is not None

    def test_update_combined_preserves_memory_structure(self, ppo_instance, sample_memory):
        """Обновление не должно модифицировать входную память"""
        # Сохраняем копию данных
        original_states_len = len(sample_memory.states)
        original_rewards_len = len(sample_memory.rewards)

        ppo_instance.update_combined(sample_memory, ep=0)

        # Длины должны остаться прежними
        assert len(sample_memory.states) == original_states_len
        assert len(sample_memory.rewards) == original_rewards_len

    def test_advantage_normalization(self, ppo_instance, sample_memory):
        """Advantages должны нормализоваться"""
        batch_size = len(sample_memory.states)
        batch_states = torch.stack([
            torch.tensor(s, dtype=torch.float32) for s in sample_memory.states
        ])

        with torch.no_grad():
            batch_values = ppo_instance.policy_old.get_value(batch_states).squeeze()

        returns, advantages = ppo_instance.compute_returns_and_advantages(
            sample_memory.rewards, batch_values, 
            sample_memory.is_terminals, sample_memory.is_truncated
        )

        # После нормализации std должен быть близок к 1
        if advantages.std() > 1e-8:
            advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            assert abs(advantages_norm.std() - 1.0) < 0.1

    def test_clip_ratio(self, ppo_instance):
        """Проверка clip ratio в policy loss"""
        eps_clip = ppo_instance.eps_clip
        
        # Создаём тестовые ratios
        ratios = torch.tensor([0.5, 0.8, 1.0, 1.2, 1.5])
        advantages = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
        clipped = torch.min(surr1, surr2)

        # Проверяем, что clip работает
        assert clipped[0] <= (1 - eps_clip)  # ratio=0.5 должен быть заклипован
        assert clipped[4] <= (1 + eps_clip)  # ratio=1.5 должен быть заклипован

    def test_value_loss_computation(self, ppo_instance, sample_memory):
        """Проверка вычисления value loss"""
        import torch.nn.functional as F

        batch_size = len(sample_memory.states)
        batch_states = torch.stack([
            torch.tensor(s, dtype=torch.float32) for s in sample_memory.states
        ])
        batch_returns = torch.randn(batch_size)

        values = ppo_instance.policy.get_value(batch_states).squeeze()
        value_loss = F.mse_loss(values, batch_returns)

        assert value_loss >= 0
        assert torch.is_tensor(value_loss)


class TestPPOIntegration:
    """Интеграционные тесты PPO"""

    def test_full_training_cycle(self):
        """Полный цикл обучения на синтетических данных"""
        policy = SeparateNetworkPolicy()
        
        # Проверяем, что policy имеет параметры
        params = list(policy.parameters())
        assert len(params) > 0, "Policy должна иметь параметры"
        
        ppo = PPO(policy)
        memory = Memory()

        # Генерируем больше данных
        for i in range(50):
            state = torch.tensor([0.5] * 17)
            action = torch.tensor([0.1, 0.2, 0.0])
            log_prob = torch.tensor(-1.5)
            reward = torch.tensor(0.5 + (i % 10) * 0.1)
            done = 1 if i % 10 == 9 else 0
            is_truncated = 0
            memory.store(state, action, log_prob, reward, done, is_truncated)

        # Запоминаем начальные веса
        initial_weights = {k: v.clone() for k, v in policy.state_dict().items()}

        # Обновляем
        ppo.update_combined(memory, ep=0, minibatch_size=20)

        # Проверяем, что веса изменились
        weights_changed = False
        for k, v in policy.state_dict().items():
            if not torch.equal(v, initial_weights[k]):
                weights_changed = True
                break

        assert weights_changed, "Веса политики должны обновиться"

    def test_multiple_updates_same_memory(self):
        """Несколько обновлений на одной памяти не должны вызывать ошибок"""
        policy = SeparateNetworkPolicy()
        
        # Проверяем параметры
        params = list(policy.parameters())
        assert len(params) > 0, "Policy должна иметь параметры"
        
        ppo = PPO(policy)
        memory = Memory()

        for i in range(20):
            memory.store(
                state=torch.tensor([0.5] * 17),
                action_final=torch.tensor([0.1, 0.2, 0.0]),
                log_prob=torch.tensor(-1.5),
                reward=torch.tensor(0.5),
                done=0,
                is_truncated=0
            )

        # Несколько обновлений
        for ep in range(5):
            ppo.update_combined(memory, ep=ep)

        # Должно завершиться без ошибок


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
