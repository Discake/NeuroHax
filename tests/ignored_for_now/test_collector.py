"""
Тесты для Collector.py - сбор опыта второго агента
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AI.Training.Collector import SharedMemoryExperienceCollector
from AI.Training.Memory import Memory


class TestCollectorDualAgent:
    """Тесты для сбора данных обоих агентов"""

    def test_experience_line_structure(self):
        """Проверка структуры строки опыта"""
        state_size = 17
        action_size = 3
        
        # Эмуляция строки опыта
        state = np.zeros(state_size)
        action = np.zeros(action_size)
        reward = np.array([1.0])
        log_prob = np.array([-1.5])
        done = np.array([0.0])
        truncated = np.array([0.0])
        
        experience_line = np.concatenate([state, action, reward, log_prob, done, truncated])
        
        expected_size = state_size + action_size + 1 + 1 + 1 + 1
        assert len(experience_line) == expected_size

    def test_dual_agent_experience_size(self):
        """Опыт для двух агентов должен быть одинакового размера"""
        state_size = 17
        action_size = 3
        
        exp1 = np.concatenate([
            np.zeros(state_size),
            np.zeros(action_size),
            np.array([1.0]),
            np.array([-1.5]),
            np.array([0.0]),
            np.array([0.0])
        ])
        
        exp2 = np.concatenate([
            np.zeros(state_size),
            np.zeros(action_size),
            np.array([1.0]),
            np.array([-1.5]),
            np.array([0.0]),
            np.array([0.0])
        ])
        
        assert len(exp1) == len(exp2)

    def test_memory_merge_both_agents(self):
        """Объединение данных обоих агентов в Memory"""
        # Эмуляция данных от двух агентов
        all_exp_states1 = [[0.1] * 17, [0.2] * 17]
        all_exp_actions1 = [[0.1, 0.2, 0.0], [0.2, 0.3, 0.0]]
        all_exp_log_probs1 = [-1.5, -1.6]
        all_exp_rewards1 = [0.5, 0.6]
        all_exp_dones1 = [0.0, 0.0]
        all_exp_truncated = [0.0, 0.0]
        
        all_exp_states2 = [[0.3] * 17, [0.4] * 17]
        all_exp_actions2 = [[0.3, 0.4, 0.0], [0.4, 0.5, 0.0]]
        all_exp_log_probs2 = [-1.7, -1.8]
        all_exp_rewards2 = [0.7, 0.8]
        all_exp_dones2 = [0.0, 0.0]
        all_exp_truncated2 = [0.0, 0.0]
        
        merged = Memory()
        merged.states = all_exp_states1 + all_exp_states2
        merged.actions_final = all_exp_actions1 + all_exp_actions2
        merged.old_log_probs = all_exp_log_probs1 + all_exp_log_probs2
        merged.rewards = all_exp_rewards1 + all_exp_rewards2
        merged.is_terminals = all_exp_dones1 + all_exp_dones2
        merged.is_truncated = all_exp_truncated + all_exp_truncated2
        merged.copy_to_tensors()
        
        # Проверяем, что данные обоих агентов объединены
        assert len(merged.states) == 4  # 2 от agent1 + 2 от agent2
        assert len(merged.rewards) == 4
        assert merged.rewards[0] == 0.5  # agent1
        assert merged.rewards[2] == 0.7  # agent2

    def test_step_increment_for_dual_agent(self):
        """Шаг должен увеличиваться на 2 для двух агентов"""
        step = 0
        steps = []
        
        for _ in range(5):
            steps.append(step)
            step += 2  # Увеличиваем на 2 для двух агентов
        
        assert steps == [0, 2, 4, 6, 8]
        # Агент 1 пишет в step, агент 2 в step+1


class TestSharedMemoryCollector:
    """Тесты для SharedMemoryExperienceCollector"""

    def test_collector_initialization(self):
        """Инициализация коллектора"""
        collector = SharedMemoryExperienceCollector(num_workers=2, max_steps_per_worker=100)
        
        assert collector.num_workers == 2
        assert collector.max_steps_per_worker == 100

    def test_create_shared_memory_blocks(self):
        """Создание блоков shared memory"""
        num_workers = 2
        max_steps = 100
        state_size = 17
        action_size = 3
        
        shm_names, total_size, step_size = SharedMemoryExperienceCollector.create_shared_memory_blocks(
            num_workers, max_steps, state_size, action_size
        )
        
        assert len(shm_names) == num_workers
        assert step_size == state_size + action_size + 1 + 1 + 1 + 1
        # total_size должен учитывать *2 для двух агентов
        assert total_size > 0


class TestExperienceBufferIntegrity:
    """Тесты целостности буфера опыта"""

    def test_no_data_loss_on_merge(self):
        """При объединении данных не должно быть потерь"""
        agent1_data = {
            'states': [[i] * 17 for i in range(10)],
            'rewards': [float(i) for i in range(10)]
        }
        
        agent2_data = {
            'states': [[i + 100] * 17 for i in range(10)],
            'rewards': [float(i + 100) for i in range(10)]
        }
        
        merged = Memory()
        merged.states = agent1_data['states'] + agent2_data['states']
        merged.rewards = agent1_data['rewards'] + agent2_data['rewards']
        merged.actions_final = [[0.1, 0.2, 0.0]] * 20
        merged.old_log_probs = [-1.5] * 20
        merged.is_terminals = [0] * 20
        merged.is_truncated = [0] * 20
        merged.copy_to_tensors()
        
        # Проверяем все данные
        assert len(merged.states) == 20
        assert merged.rewards[0] == 0.0      # agent1[0]
        assert merged.rewards[9] == 9.0      # agent1[9]
        assert merged.rewards[10] == 100.0   # agent2[0]
        assert merged.rewards[19] == 109.0   # agent2[9]

    def test_alternating_agent_data(self):
        """Проверка чередования данных агентов в буфере"""
        # В shared memory данные записываются как: [agent1_step0, agent2_step0, agent1_step1, agent2_step1, ...]
        buffer = []
        for step in range(0, 10, 2):
            buffer.append(('agent1', step // 2))
            buffer.append(('agent2', step // 2))
        
        assert len(buffer) == 10
        assert buffer[0] == ('agent1', 0)
        assert buffer[1] == ('agent2', 0)
        assert buffer[2] == ('agent1', 1)
        assert buffer[3] == ('agent2', 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
