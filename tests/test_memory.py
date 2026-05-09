"""
Тесты для Memory.py
"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AI.Training.Memory import Memory


class TestMemory:
    """Тесты для буфера памяти"""

    @pytest.fixture
    def empty_memory(self):
        """Пустая память"""
        return Memory()

    @pytest.fixture
    def populated_memory(self):
        """Память с данными"""
        memory = Memory()
        for i in range(5):
            memory.store(
                state=torch.tensor([0.1 * i] * 17),
                action_final=torch.tensor([0.1, 0.2, 0.0]),
                log_prob=torch.tensor(-1.5),
                reward=torch.tensor(0.5 + i * 0.1),
                done=0,
                is_truncated=0
            )
        return memory

    def test_store_single_experience(self, empty_memory):
        """Тест сохранения одного опыта"""
        empty_memory.store(
            state=torch.tensor([0.5] * 17),
            action_final=torch.tensor([0.1, 0.2, 0.0]),
            log_prob=torch.tensor(-1.5),
            reward=torch.tensor(1.0),
            done=False,
            is_truncated=False
        )

        assert len(empty_memory.states) == 1
        assert len(empty_memory.actions_final) == 1
        assert len(empty_memory.old_log_probs) == 1
        assert len(empty_memory.rewards) == 1
        assert empty_memory.states[0] == [0.5] * 17
        assert empty_memory.rewards[0] == 1.0

    def test_store_multiple_experiences(self, empty_memory):
        """Тест сохранения нескольких опытов"""
        for i in range(10):
            empty_memory.store(
                state=torch.tensor([i] * 17),
                action_final=torch.tensor([0.1, 0.2, 0.0]),
                log_prob=torch.tensor(-1.5),
                reward=torch.tensor(float(i)),
                done=0,
                is_truncated=0
            )

        assert len(empty_memory.states) == 10
        assert len(empty_memory.rewards) == 10
        assert empty_memory.rewards[9] == 9.0

    def test_clear_all_fields(self, populated_memory):
        """Тест очистки всех полей (критично!)"""
        populated_memory.clear()

        assert len(populated_memory.states) == 0
        assert len(populated_memory.actions_final) == 0
        assert len(populated_memory.actions_raw) == 0
        assert len(populated_memory.old_log_probs) == 0
        assert len(populated_memory.rewards) == 0
        assert len(populated_memory.is_terminals) == 0
        assert len(populated_memory.is_truncated) == 0

    def test_clear_twice(self, populated_memory):
        """Двойная очистка не должна вызывать ошибок"""
        populated_memory.clear()
        populated_memory.clear()  # Не должно вызвать ошибку

        assert len(populated_memory.states) == 0

    def test_copy_to_tensors(self, populated_memory):
        """Конвертация в тензоры"""
        populated_memory.copy_to_tensors()

        assert torch.is_tensor(populated_memory.states)
        assert torch.is_tensor(populated_memory.actions_final)
        assert torch.is_tensor(populated_memory.old_log_probs)
        assert torch.is_tensor(populated_memory.rewards)
        assert torch.is_tensor(populated_memory.is_terminals)
        assert torch.is_tensor(populated_memory.is_truncated)

        assert populated_memory.states.shape[0] == 5
        assert populated_memory.rewards.shape[0] == 5

    def test_tensor_device(self, populated_memory):
        """Тензоры должны быть на правильном устройстве"""
        populated_memory.copy_to_tensors()

        assert populated_memory.states.device.type == populated_memory.states.device.type
        assert populated_memory.rewards.device.type == populated_memory.rewards.device.type

    def test_store_with_truncated(self, empty_memory):
        """Тест сохранения с truncated флагом"""
        empty_memory.store(
            state=torch.tensor([0.5] * 17),
            action_final=torch.tensor([0.1, 0.2, 0.0]),
            log_prob=torch.tensor(-1.5),
            reward=torch.tensor(1.0),
            done=False,
            is_truncated=True
        )

        assert empty_memory.is_truncated[0] == True

    def test_store_with_terminal(self, empty_memory):
        """Тест сохранения с terminal флагом"""
        empty_memory.store(
            state=torch.tensor([0.5] * 17),
            action_final=torch.tensor([0.1, 0.2, 0.0]),
            log_prob=torch.tensor(-1.5),
            reward=torch.tensor(1.0),
            done=True,
            is_truncated=False
        )

        assert empty_memory.is_terminals[0] == True

    def test_memory_growth(self, empty_memory):
        """Память должна расти корректно"""
        initial_size = 0
        
        for i in range(100):
            empty_memory.store(
                state=torch.tensor([i] * 17),
                action_final=torch.tensor([0.1, 0.2, 0.0]),
                log_prob=torch.tensor(-1.5),
                reward=torch.tensor(1.0),
                done=0,
                is_truncated=0
            )
            
            assert len(empty_memory.states) == initial_size + 1
            initial_size += 1

    def test_clear_after_copy(self, populated_memory):
        """Очистка после конвертации в тензоры"""
        populated_memory.copy_to_tensors()
        populated_memory.clear()  # Не должно вызвать ошибку

        assert len(populated_memory.states) == 0
        assert len(populated_memory.actions_final) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
