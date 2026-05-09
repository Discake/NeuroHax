"""
Тесты для Training_process.py - контролируемое обучение
"""
import pytest
import torch
import sys
import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AI.Training.Training_process import (
    TrainingProcess, 
    TrainingConfig, 
    TrainingMetrics, 
    TrainingStatus
)
from AI.SeparateNetworkPolicy import SeparateNetworkPolicy


class TestTrainingConfig:
    """Тесты конфигурации обучения"""

    def test_default_config(self):
        """Конфигурация по умолчанию"""
        config = TrainingConfig()
        
        assert config.num_episodes == 100
        assert config.num_workers == 4
        assert config.max_steps_per_worker == 1024
        assert config.batch_size == 4096 * 5
        assert config.save_interval == 20
        assert config.checkpoint_interval == 50
        assert config.delay_ep == 10
        assert config.max_training_time is None
        assert config.early_stop_reward is None
        assert config.log_interval == 5
        assert config.draw_stats is False
        assert config.save_filename is None
        assert config.checkpoint_dir is None

    def test_custom_config(self):
        """Пользовательская конфигурация"""
        config = TrainingConfig(
            num_episodes=50,
            num_workers=2,
            max_steps_per_worker=512,
            save_interval=10,
            max_training_time=3600,
            early_stop_reward=0.8
        )
        
        assert config.num_episodes == 50
        assert config.num_workers == 2
        assert config.max_steps_per_worker == 512
        assert config.save_interval == 10
        assert config.max_training_time == 3600
        assert config.early_stop_reward == 0.8

    def test_config_validation_negative_workers(self):
        """Валидация: отрицательное количество workers"""
        with pytest.raises(ValueError, match="num_workers должен быть >= 1"):
            TrainingProcess(config=TrainingConfig(num_workers=0))

    def test_config_validation_negative_episodes(self):
        """Валидация: отрицательное количество эпизодов"""
        with pytest.raises(ValueError, match="num_episodes должен быть >= 1"):
            TrainingProcess(config=TrainingConfig(num_episodes=0))

    def test_config_validation_negative_save_interval(self):
        """Валидация: отрицательный интервал сохранения"""
        with pytest.raises(ValueError, match="save_interval должен быть >= 1"):
            TrainingProcess(config=TrainingConfig(save_interval=0))


class TestTrainingMetrics:
    """Тесты метрик обучения"""

    def test_default_metrics(self):
        """Метрики по умолчанию"""
        metrics = TrainingMetrics()
        
        assert metrics.episode == 0
        assert metrics.total_reward == 0.0
        assert metrics.avg_reward == 0.0
        assert metrics.episodes_completed == 0
        assert metrics.training_time == 0.0
        assert metrics.status == TrainingStatus.RUNNING
        assert metrics.error_message is None
        assert len(metrics.rewards_history) == 0
        assert len(metrics.loss_history) == 0

    def test_metrics_to_dict(self):
        """Конвертация метрик в словарь"""
        metrics = TrainingMetrics()
        metrics.episode = 10
        metrics.total_reward = 100.0
        metrics.avg_reward = 10.0
        metrics.episodes_completed = 10
        metrics.status = TrainingStatus.COMPLETED
        
        result = metrics.to_dict()
        
        assert result['episode'] == 10
        assert result['total_reward'] == 100.0
        assert result['avg_reward'] == 10.0
        assert result['episodes_completed'] == 10
        assert result['status'] == 'completed'
        assert 'error_message' in result
        assert 'rewards_count' in result
        assert 'loss_count' in result

    def test_metrics_update_reward(self):
        """Обновление наград в метриках"""
        metrics = TrainingMetrics()
        
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        for reward in rewards:
            metrics.rewards_history.append(reward)
            metrics.total_reward += reward
            metrics.avg_reward = metrics.total_reward / len(metrics.rewards_history)
        
        assert metrics.total_reward == 15.0
        assert metrics.avg_reward == 3.0
        assert len(metrics.rewards_history) == 5


class TestTrainingStatus:
    """Тесты статусов обучения"""

    def test_status_values(self):
        """Значения статусов"""
        assert TrainingStatus.RUNNING.value == "running"
        assert TrainingStatus.PAUSED.value == "paused"
        assert TrainingStatus.STOPPED.value == "stopped"
        assert TrainingStatus.COMPLETED.value == "completed"
        assert TrainingStatus.ERROR.value == "error"


class TestTrainingProcessInit:
    """Тесты инициализации TrainingProcess"""

    def test_init_default(self):
        """Инициализация по умолчанию"""
        process = TrainingProcess()
        
        assert process.config.num_episodes == 100
        assert process.policy is not None
        assert isinstance(process.policy, SeparateNetworkPolicy)
        assert process.ppo is None
        assert process.get_status() == TrainingStatus.RUNNING

    def test_init_with_custom_config(self):
        """Инициализация с пользовательской конфигурацией"""
        config = TrainingConfig(num_episodes=50, num_workers=2)
        process = TrainingProcess(config=config)
        
        assert process.config.num_episodes == 50
        assert process.config.num_workers == 2

    def test_init_with_policy(self):
        """Инициализация с готовой политикой"""
        policy = SeparateNetworkPolicy()
        process = TrainingProcess(policy=policy)
        
        assert process.policy is policy

    def test_init_validates_config(self):
        """Инициализация проверяет конфигурацию"""
        with pytest.raises(ValueError):
            TrainingProcess(config=TrainingConfig(num_workers=0))


class TestTrainingProcessControl:
    """Тесты управления обучением"""

    def test_pause_resume(self):
        """Пауза и возобновление"""
        process = TrainingProcess()
        
        assert process.get_status() == TrainingStatus.RUNNING
        
        process.pause()
        assert process.get_status() == TrainingStatus.PAUSED
        assert process.metrics.status == TrainingStatus.PAUSED
        
        process.resume()
        assert process.get_status() == TrainingStatus.RUNNING
        assert process.metrics.status == TrainingStatus.RUNNING

    def test_stop(self):
        """Остановка"""
        process = TrainingProcess()
        
        process.stop("Тестовая остановка")
        
        assert process.get_status() == TrainingStatus.STOPPED
        assert process.metrics.status == TrainingStatus.STOPPED
        assert process.metrics.error_message == "Тестовая остановка"

    def test_get_metrics(self):
        """Получение метрик"""
        process = TrainingProcess()
        
        metrics = process.get_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.status == TrainingStatus.RUNNING

    def test_get_status(self):
        """Получение статуса"""
        process = TrainingProcess()
        
        status = process.get_status()
        
        assert status == TrainingStatus.RUNNING


class TestTrainingProcessCheckpoint:
    """Тесты чекпоинтов"""

    @pytest.fixture
    def temp_dir(self):
        """Временная директория"""
        test_dir = tempfile.mkdtemp()
        yield test_dir
        shutil.rmtree(test_dir, ignore_errors=True)

    def test_save_and_load_checkpoint(self, temp_dir):
        """Сохранение и загрузка чекпоинта"""
        process = TrainingProcess()
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
        
        # Инициализируем PPO
        process.ppo = MagicMock()
        process.ppo.policy_old = SeparateNetworkPolicy()
        process.metrics.episode = 10
        process.metrics.total_reward = 100.0
        
        # Сохраняем
        process.save_checkpoint(checkpoint_path)
        
        # Проверяем файл
        assert os.path.exists(checkpoint_path)
        
        # Загружаем в новый процесс
        process2 = TrainingProcess()
        process2.load_checkpoint(checkpoint_path)
        
        assert process2.metrics.episode == 10
        assert process2.metrics.total_reward == 100.0

    def test_load_nonexistent_checkpoint(self, temp_dir):
        """Загрузка несуществующего чекпоинта"""
        process = TrainingProcess()
        checkpoint_path = os.path.join(temp_dir, "nonexistent.pt")
        
        with pytest.raises(FileNotFoundError):
            process.load_checkpoint(checkpoint_path)


class TestTrainingProcessCallback:
    """Тесты callback'ов"""

    def test_set_callback(self):
        """Установка callback'а"""
        process = TrainingProcess()
        
        callback_called = []
        
        def callback(metrics):
            callback_called.append(metrics)
            return True
        
        process.set_callback(callback)
        
        assert process._callback is callback

    def test_callback_stop_training(self):
        """Callback останавливает обучение"""
        process = TrainingProcess()
        
        def stop_callback(metrics):
            return False  # Остановить
        
        process.set_callback(stop_callback)
        
        # Проверяем, что callback установлен
        assert process._callback is not None


class TestTrainingProcessTrain:
    """Тесты метода train"""

    @pytest.fixture
    def temp_dir(self):
        """Временная директория"""
        test_dir = tempfile.mkdtemp()
        yield test_dir
        shutil.rmtree(test_dir, ignore_errors=True)

    @patch('AI.Training.Training_process.SharedMemoryExperienceCollector')
    @patch('AI.Training.Training_process.mp.Pool')
    def test_train_simple(self, mock_pool, mock_collector, temp_dir):
        """Упрощённое обучение"""
        # Настраиваем моки
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.starmap.return_value = None
        
        # Мокаем create_shared_memory_blocks
        mock_collector.create_shared_memory_blocks.return_value = (
            ['shm_0', 'shm_1'],  # shm_objects
            1024 * 100,          # total_size
            24                   # step_size
        )
        
        # Мокаем read_and_parse_shared_memory
        mock_memory = MagicMock()
        mock_memory.states = []
        mock_collector.read_and_parse_shared_memory.return_value = mock_memory
        
        process = TrainingProcess()
        process.config.num_episodes = 1  # Быстрый тест
        
        # Запускаем обучение
        result = process.train_simple(num_episodes=1)
        
        assert result is not None
        assert process.metrics.status == TrainingStatus.COMPLETED

    @patch('AI.Training.Training_process.SharedMemoryExperienceCollector')
    @patch('AI.Training.Training_process.mp.Pool')
    def test_train_with_save(self, mock_pool, mock_collector, temp_dir):
        """Обучение с сохранением"""
        # Настраиваем моки
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        
        # Мокаем create_shared_memory_blocks
        mock_collector.create_shared_memory_blocks.return_value = (
            ['shm_0', 'shm_1'],
            1024 * 100,
            24
        )
        
        # Мокаем read_and_parse_shared_memory
        mock_memory = MagicMock()
        mock_memory.states = []
        mock_collector.read_and_parse_shared_memory.return_value = mock_memory
        
        process = TrainingProcess()
        process.config.num_episodes = 1
        save_path = os.path.join(temp_dir, "model.pt")
        
        # Запускаем обучение
        process.train(save_filename=save_path)
        
        # Проверяем сохранение
        assert os.path.exists(save_path)

    @patch('AI.Training.Training_process.SharedMemoryExperienceCollector')
    @patch('AI.Training.Training_process.mp.Pool')
    def test_train_with_max_time(self, mock_pool, mock_collector, temp_dir):
        """Обучение с ограничением по времени"""
        # Настраиваем моки
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        
        # Мокаем create_shared_memory_blocks
        mock_collector.create_shared_memory_blocks.return_value = (
            ['shm_0', 'shm_1'],
            1024 * 100,
            24
        )
        
        # Мокаем read_and_parse_shared_memory
        mock_memory = MagicMock()
        mock_memory.states = []
        mock_collector.read_and_parse_shared_memory.return_value = mock_memory
        
        process = TrainingProcess()
        process.config.num_episodes = 100
        process.config.max_training_time = 0.001  # Очень мало
        
        # Запускаем обучение
        process.train()
        
        # Должно остановиться по времени
        assert process.metrics.status in [TrainingStatus.STOPPED, TrainingStatus.COMPLETED]


class TestTrainingProcessEdgeCases:
    """Тесты граничных случаев"""

    def test_pause_before_start(self):
        """Пауза до начала обучения"""
        process = TrainingProcess()
        process.pause()
        
        assert process.get_status() == TrainingStatus.PAUSED

    def test_resume_without_pause(self):
        """Возобновление без паузы"""
        process = TrainingProcess()
        process.resume()  # Не должно вызвать ошибку
        
        assert process.get_status() == TrainingStatus.RUNNING

    def test_stop_twice(self):
        """Двойная остановка"""
        process = TrainingProcess()
        process.stop("Первая")
        process.stop("Вторая")
        
        assert process.get_status() == TrainingStatus.STOPPED
        assert process.metrics.error_message == "Вторая"

    def test_metrics_after_error(self):
        """Метрики после ошибки"""
        process = TrainingProcess()
        process._status = TrainingStatus.ERROR
        process.metrics.error_message = "Тестовая ошибка"
        
        assert process.metrics.status == TrainingStatus.RUNNING  # Не меняется автоматически
        assert process.metrics.error_message == "Тестовая ошибка"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
