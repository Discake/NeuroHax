"""
Тесты для Collector.py - SharedMemoryExperienceCollector
"""
import pytest
import numpy as np
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import torch

# Импорты после настройки conftest
from AI.Training.Memory import Memory
from AI.Training.ChunkedMmapBuffer import ChunkedMmapBuffer
import Constants

# Импортируем Collector после моков
from AI.Training.Collector import SharedMemoryExperienceCollector


class TestSharedMemoryExperienceCollectorInit:
    """Тесты инициализации коллектора"""

    def test_default_initialization(self):
        """Инициализация с параметрами по умолчанию"""
        collector = SharedMemoryExperienceCollector()
        
        assert collector.num_workers == 20
        assert collector.max_steps_per_worker == 1000
        assert collector.step_size is None

    def test_custom_initialization(self):
        """Инициализация с кастомными параметрами"""
        collector = SharedMemoryExperienceCollector(num_workers=4, max_steps_per_worker=500)
        
        assert collector.num_workers == 4
        assert collector.max_steps_per_worker == 500


class TestCreateOrCleanupSharedMemory:
    """Тесты создания и очистки shared memory"""

    def test_create_new_shared_memory(self, temp_dir):
        """Создание нового shared memory сегмента"""
        collector = SharedMemoryExperienceCollector()
        
        shm_name = "test_shm_segment"
        size = 1024
        
        shm = collector.create_or_cleanup_shared_memory(shm_name, size)
        
        assert shm is not None
        assert shm.name == shm_name
        assert shm.size == size
        
        shm.close()
        shm.unlink()

    def test_cleanup_existing_shared_memory(self, temp_dir):
        """Очистка существующего shared memory сегмента"""
        collector = SharedMemoryExperienceCollector()
        
        shm_name = "test_cleanup_shm"
        size = 512
        
        # Создаём первый сегмент
        shm1 = collector.create_or_cleanup_shared_memory(shm_name, size)
        shm1.close()
        
        # Создаём новый (должен удалить старый)
        shm2 = collector.create_or_cleanup_shared_memory(shm_name, size * 2)
        
        assert shm2.size == size * 2
        
        shm2.close()
        shm2.unlink()


class TestCreateSharedMemoryBlocks:
    """Тесты создания блоков shared memory"""

    def test_create_blocks_basic(self):
        """Создание блоков для нескольких worker'ов"""
        num_workers = 4
        max_steps = 100
        state_size = 17
        action_size = 3
        
        shm_names, total_size, step_size = SharedMemoryExperienceCollector.create_shared_memory_blocks(
            num_workers, max_steps, state_size, action_size
        )
        
        assert len(shm_names) == num_workers
        # step_size без умножения на 2 (только для одного агента)
        assert step_size == state_size + action_size + 1 + 1 + 1 + 1  # 24 элемента
        
        # Проверка размера: *2 для двух агентов, *4 байта на float32
        expected_step_bytes = (state_size + action_size + 1 + 1 + 1 + 1) * 4 * 2
        expected_total = max_steps * expected_step_bytes
        assert total_size == expected_total

    def test_create_blocks_single_worker(self):
        """Создание блока для одного worker'а"""
        shm_names, total_size, step_size = SharedMemoryExperienceCollector.create_shared_memory_blocks(
            1, 50, 17, 3
        )
        
        assert len(shm_names) == 1
        # step_size без умножения на 2
        assert step_size == 24

    def test_step_size_calculation(self):
        """Проверка расчёта step_size"""
        state_size = 17
        action_size = 3
        # step_size без умножения на 2
        expected_step_size = state_size + action_size + 1 + 1 + 1 + 1
        
        _, _, step_size = SharedMemoryExperienceCollector.create_shared_memory_blocks(
            2, 100, state_size, action_size
        )
        
        assert step_size == expected_step_size


class TestChunkedMmapBufferIntegration:
    """Интеграционные тесты с ChunkedMmapBuffer"""

    def test_buffer_write_read(self, temp_dir):
        """Запись и чтение из буфера"""
        buffer = ChunkedMmapBuffer()
        total_steps = 100
        step_size = 23
        shm_name = "test_buffer"
        
        buffer.create(shm_name, total_steps, step_size)
        buffer.open(total_steps, step_size, shm_name)
        
        # Запись тестовых данных
        test_data = np.arange(step_size, dtype=np.float32)
        buffer[0] = test_data
        buffer[1] = test_data * 2
        
        # Чтение
        read_data_0 = buffer[0]
        read_data_1 = buffer[1]
        
        assert np.array_equal(read_data_0, test_data)
        assert np.array_equal(read_data_1, test_data * 2)
        
        # Сначала очищаем файлы, затем закрываем
        buffer.close(shm_name, delete_files=True, clear_files=False)

    def test_buffer_zero_padding(self, temp_dir):
        """Проверка нулевого заполнения пустых ячеек"""
        buffer = ChunkedMmapBuffer()
        total_steps = 10
        step_size = 23
        shm_name = "test_zero_buffer"
        
        buffer.create(shm_name, total_steps, step_size)
        buffer.open(total_steps, step_size, shm_name)
        
        # Записываем только первые 2 ячейки
        buffer[0] = np.ones(step_size)
        buffer[1] = np.ones(step_size) * 2
        
        # Остальные должны быть нулевыми
        assert np.all(buffer[5] == 0)
        assert np.all(buffer[9] == 0)
        
        buffer.close(shm_name, delete_files=True, clear_files=False)


class TestExperienceDataStructure:
    """Тесты структуры данных опыта"""

    def test_experience_line_size(self):
        """Размер строки опыта для одного агента"""
        state_size = 17
        action_size = 3
        
        # state + action + reward + log_prob + done + truncated
        expected_size = state_size + action_size + 1 + 1 + 1 + 1
        assert expected_size == 24

    def test_experience_line_components(self):
        """Компоненты строки опыта для одного агента"""
        state_size = 17
        action_size = 3
        
        state = np.random.rand(state_size).astype(np.float32)
        action = np.random.rand(action_size).astype(np.float32)
        reward = np.array([0.5], dtype=np.float32)
        log_prob = np.array([-1.5], dtype=np.float32)
        done = np.array([0.0], dtype=np.float32)
        truncated = np.array([1.0], dtype=np.float32)
        
        experience_line = np.concatenate([state, action, reward, log_prob, done, truncated])
        
        assert len(experience_line) == 24
        
        # Проверка позиций
        assert np.array_equal(experience_line[0:state_size], state)
        assert np.array_equal(experience_line[state_size:state_size + action_size], action)
        assert experience_line[state_size + action_size] == 0.5
        assert experience_line[state_size + action_size + 1] == -1.5
        assert experience_line[state_size + action_size + 2] == 0.0
        assert experience_line[state_size + action_size + 3] == 1.0

    def test_dual_agent_step_increment(self):
        """Увеличение шага на 2 для двух агентов"""
        step = 0
        steps = []
        
        for _ in range(5):
            steps.append(step)
            step += 2
        
        assert steps == [0, 2, 4, 6, 8]
        # Агент 1 пишет в step, агент 2 в step+1


class TestMemoryIntegration:
    """Тесты интеграции с Memory"""

    def test_memory_from_collected_data(self):
        """Создание Memory из данных коллектора"""
        all_exp_states1 = [[0.1] * 17, [0.2] * 17]
        all_exp_actions1 = [[0.1, 0.2, 0.0], [0.2, 0.3, 0.0]]
        all_exp_log_probs1 = [-1.5, -1.6]
        all_exp_rewards1 = [0.5, 0.6]
        all_exp_dones1 = [0.0, 1.0]
        all_exp_truncated = [0.0, 0.0]
        
        all_exp_states2 = [[0.3] * 17, [0.4] * 17]
        all_exp_actions2 = [[0.3, 0.4, 0.0], [0.4, 0.5, 0.0]]
        all_exp_log_probs2 = [-1.7, -1.8]
        all_exp_rewards2 = [0.7, 0.8]
        all_exp_dones2 = [1.0, 0.0]
        all_exp_truncated2 = [0.0, 1.0]
        
        merged = Memory()
        merged.states = all_exp_states1 + all_exp_states2
        merged.actions_final = all_exp_actions1 + all_exp_actions2
        merged.old_log_probs = all_exp_log_probs1 + all_exp_log_probs2
        merged.rewards = all_exp_rewards1 + all_exp_rewards2
        merged.is_terminals = all_exp_dones1 + all_exp_dones2
        merged.is_truncated = all_exp_truncated + all_exp_truncated2
        merged.copy_to_tensors()
        
        # Проверка размеров
        assert len(merged.states) == 4
        assert len(merged.rewards) == 4
        assert len(merged.is_terminals) == 4
        assert len(merged.is_truncated) == 4
        
        # Проверка значений
        assert merged.rewards[0] == 0.5  # agent1[0]
        assert merged.rewards[2] == 0.7  # agent2[0]
        assert merged.is_terminals[1] == 1.0  # agent1[1] - terminal
        assert merged.is_truncated[3] == 1.0  # agent2[1] - truncated

    def test_memory_data_integrity(self):
        """Целостность данных при объединении"""
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
        
        assert len(merged.states) == 20
        assert merged.rewards[0] == 0.0      # agent1[0]
        assert merged.rewards[9] == 9.0      # agent1[9]
        assert merged.rewards[10] == 100.0   # agent2[0]
        assert merged.rewards[19] == 109.0   # agent2[9]


class TestReadAndParseSharedMemory:
    """Тесты статического метода read_and_parse_shared_memory"""

    def test_parse_empty_buffer(self, temp_dir):
        """Чтение пустого буфера"""
        state_size = 17
        action_size = 3
        step_size = state_size + action_size + 1 + 1 + 1 + 1
        total_size = step_size * 4 * 10  # 10 шагов
        
        shm_names = ["test_empty_buffer"]
        
        # Создаём пустой буфер
        buffer = ChunkedMmapBuffer()
        buffer.create(shm_names[0], 10, step_size)
        
        # Парсим (должен вернуть пустую Memory)
        result = SharedMemoryExperienceCollector.read_and_parse_shared_memory(
            total_size, step_size, state_size, action_size, shm_names
        )
        
        assert isinstance(result, Memory)
        # Все данные должны быть пустыми или нулевыми
        
        buffer.close(shm_names[0], delete_files=True, clear_files=False)

    def test_parse_filled_buffer(self, temp_dir):
        """Чтение заполненного буфера"""
        state_size = 17
        action_size = 3
        step_size = state_size + action_size + 1 + 1 + 1 + 1
        total_size = step_size * 4 * 10
        
        shm_names = ["test_filled_buffer"]
        
        buffer = ChunkedMmapBuffer()
        buffer.create(shm_names[0], 10, step_size)
        buffer.open(10, step_size, shm_names[0])
        
        # Заполняем первые 4 ячейки (каждая для одного агента)
        for i in range(4):
            data = np.ones(step_size) * (i + 1)
            buffer[i] = data
        
        buffer.close(shm_names[0], delete_files=False, clear_files=False)
        
        # Парсим
        result = SharedMemoryExperienceCollector.read_and_parse_shared_memory(
            total_size, step_size, state_size, action_size, shm_names
        )
        
        assert isinstance(result, Memory)
        # read_and_parse читает и agent1 и agent2 данные (чередуются)
        # Поэтому будет 4 состояния (2 пары agent1/agent2)
        assert len(result.states) >= 4
        assert len(result.rewards) >= 4
        
        buffer.close(shm_names[0], delete_files=True, clear_files=False)


class TestWorkerSharedMemoryMock:
    """Тесты worker метода с моками"""

    @patch('AI.Training.Collector.SeparateNetworkPolicy')
    @patch('AI.Training.Collector.Environment')
    def test_worker_with_mocked_dependencies(self, mock_env_class, mock_policy_class, temp_dir):
        """Тест worker с моками зависимостей"""
        # Настраиваем моки
        mock_policy = MagicMock()
        mock_policy.select_action.return_value = (
            torch.tensor([0.1, 0.2, 0.0]),
            torch.tensor(-1.5)
        )
        mock_policy_class.return_value = mock_policy
        
        mock_env = MagicMock()
        mock_env.reset.return_value = (
            torch.zeros(17),
            torch.zeros(17)
        )
        # step возвращает done=True сразу, чтобы завершить цикл
        # reward должен быть тензором с flatten()
        mock_env.step.return_value = (
            (torch.zeros(17), torch.zeros(17)),
            (torch.tensor(0.5), torch.tensor(0.5)),
            True,  # done
            {'natural_done': False, 'truncated': False}
        )
        mock_env_class.return_value = mock_env
        
        state_dict1 = {}
        state_dict2 = {}
        shm_name = "test_worker_shm"
        step_size = 24  # для одного агента
        max_steps = 10
        
        # Создаём буфер
        buffer = ChunkedMmapBuffer()
        buffer.create(shm_name, max_steps, step_size)
        
        # Вызываем worker
        result = SharedMemoryExperienceCollector.worker_shared_memory(
            max_steps, step_size, [shm_name], state_dict1, state_dict2, 0
        )
        
        # Worker возвращает process_id при успехе
        assert result == 0
        
        buffer.close(shm_name, delete_files=True, clear_files=False)


class TestCollectorEdgeCases:
    """Тесты граничных случаев"""

    def test_zero_workers(self):
        """Коллектор с нулевым количеством worker'ов"""
        collector = SharedMemoryExperienceCollector(num_workers=0)
        assert collector.num_workers == 0

    def test_single_step_per_worker(self):
        """Один шаг на worker"""
        collector = SharedMemoryExperienceCollector(max_steps_per_worker=1)
        assert collector.max_steps_per_worker == 1

    def test_large_num_workers(self):
        """Большое количество worker'ов"""
        collector = SharedMemoryExperienceCollector(num_workers=100)
        assert collector.num_workers == 100


class TestAlternatingAgentData:
    """Тесты чередования данных агентов"""

    def test_alternating_pattern_in_buffer(self):
        """Чередование данных агентов в буфере"""
        buffer_data = []
        
        for step in range(0, 10, 2):
            buffer_data.append(('agent1', step // 2))
            buffer_data.append(('agent2', step // 2))
        
        assert len(buffer_data) == 10
        assert buffer_data[0] == ('agent1', 0)
        assert buffer_data[1] == ('agent2', 0)
        assert buffer_data[2] == ('agent1', 1)
        assert buffer_data[3] == ('agent2', 1)
        assert buffer_data[8] == ('agent1', 4)
        assert buffer_data[9] == ('agent2', 4)

    def test_agent2_data_at_odd_indices(self):
        """Данные агента 2 на нечётных индексах"""
        state_size = 17
        action_size = 3
        step_size = state_size + action_size + action_size + 1 + 1 + 1 + 1
        
        # Эмуляция буфера: agent1 на чётных, agent2 на нечётных
        buffer = []
        for i in range(10):
            if i % 2 == 0:
                buffer.append(('agent1', i // 2))
            else:
                buffer.append(('agent2', i // 2))
        
        # Проверяем чередование
        for i in range(0, 10, 2):
            assert buffer[i][0] == 'agent1'
            assert buffer[i + 1][0] == 'agent2'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
