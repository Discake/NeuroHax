import torch
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List
from enum import Enum

from AI.Training.Collector import SharedMemoryExperienceCollector
from AI.Training.Memory import Memory
from AI.Training.ChunkedMmapBuffer import ChunkedMmapBuffer
from AI.Training.PPO import PPO
from AI.Training.Environment import Environment
import Constants
from AI.SeparateNetworkPolicy import SeparateNetworkPolicy

import multiprocessing as mp
from multiprocessing import shared_memory


class TrainingStatus(Enum):
    """Статус процесса обучения"""
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    num_episodes: int = 100
    num_workers: int = 4
    max_steps_per_worker: int = 1024
    batch_size: int = 4096 * 5
    save_interval: int = 20
    checkpoint_interval: int = 50
    delay_ep: int = 10  # Интервал обновления delay policy
    
    # Ограничения
    max_training_time: Optional[float] = None  # Максимальное время в секундах
    early_stop_reward: Optional[float] = None  # Ранняя остановка при достижении награды
    
    # Логирование
    log_interval: int = 5  # Логировать каждый N эпизод
    draw_stats: bool = False
    
    # Пути
    save_filename: Optional[str] = None
    checkpoint_dir: Optional[str] = None


@dataclass
class TrainingMetrics:
    """Метрики обучения"""
    episode: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    episodes_completed: int = 0
    training_time: float = 0.0
    status: TrainingStatus = TrainingStatus.RUNNING
    error_message: Optional[str] = None
    rewards_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'episode': self.episode,
            'total_reward': self.total_reward,
            'avg_reward': self.avg_reward,
            'episodes_completed': self.episodes_completed,
            'training_time': self.training_time,
            'status': self.status.value,
            'error_message': self.error_message,
            'rewards_count': len(self.rewards_history),
            'loss_count': len(self.loss_history)
        }


class TrainingProcess:
    """
    Контролируемый процесс обучения с поддержкой:
    - Pause/Resume/Stop
    - Чекпоинты
    - Метрики и логирование
    - Ранняя остановка
    - Ограничение по времени
    """
    
    def __init__(
        self, 
        env: Optional[Environment] = None,
        config: Optional[TrainingConfig] = None,
        policy: Optional[SeparateNetworkPolicy] = None
    ):
        self.config = config or TrainingConfig()
        self.env = env
        self.policy = policy or SeparateNetworkPolicy()
        
        self.ppo: Optional[PPO] = None
        self.metrics = TrainingMetrics()
        self._status = TrainingStatus.RUNNING
        self._start_time: Optional[float] = None
        self._callback: Optional[Callable[[TrainingMetrics], bool]] = None
        
        # Валидация конфигурации
        self._validate_config()
    
    def _validate_config(self):
        """Проверка корректности конфигурации"""
        if self.config.num_workers < 1:
            raise ValueError("num_workers должен быть >= 1")
        if self.config.max_steps_per_worker < 1:
            raise ValueError("max_steps_per_worker должен быть >= 1")
        if self.config.num_episodes < 1:
            raise ValueError("num_episodes должен быть >= 1")
        if self.config.save_interval < 1:
            raise ValueError("save_interval должен быть >= 1")
    
    def set_callback(self, callback: Callable[[TrainingMetrics], bool]):
        """
        Установить callback для мониторинга обучения.
        Callback получает метрики и возвращает True для продолжения, False для остановки.
        """
        self._callback = callback
    
    def pause(self):
        """Приостановить обучение"""
        self._status = TrainingStatus.PAUSED
        self.metrics.status = TrainingStatus.PAUSED
    
    def resume(self):
        """Возобновить обучение"""
        if self._status == TrainingStatus.PAUSED:
            self._status = TrainingStatus.RUNNING
            self.metrics.status = TrainingStatus.RUNNING
    
    def stop(self, reason: str = "Остановлено пользователем"):
        """Остановить обучение"""
        self._status = TrainingStatus.STOPPED
        self.metrics.status = TrainingStatus.STOPPED
        self.metrics.error_message = reason
    
    def get_metrics(self) -> TrainingMetrics:
        """Получить текущие метрики обучения"""
        return self.metrics
    
    def get_status(self) -> TrainingStatus:
        """Получить текущий статус"""
        return self._status
    
    def load_checkpoint(self, checkpoint_path: str):
        """Загрузить чекпоинт"""
        checkpoint = torch.load(checkpoint_path, map_location=Constants.device, weights_only=False)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.ppo = PPO(self.policy)
        
        if 'ppo_state_dict' in checkpoint:
            self.ppo.policy_old.load_state_dict(checkpoint['ppo_state_dict'])
        
        if 'metrics' in checkpoint:
            saved_metrics = checkpoint['metrics']
            # Восстанавливаем метрики
            self.metrics.episode = saved_metrics.episode
            self.metrics.total_reward = saved_metrics.total_reward
            self.metrics.avg_reward = saved_metrics.avg_reward
            self.metrics.episodes_completed = saved_metrics.episodes_completed
            self.metrics.training_time = saved_metrics.training_time
        
        if 'episode' in checkpoint:
            self.metrics.episode = checkpoint['episode']
        
        print(f"Чекпоинт загружен: эпизод {self.metrics.episode}")
    
    def save_checkpoint(self, checkpoint_path: str):
        """Сохранить чекпоинт"""
        checkpoint = {
            'episode': self.metrics.episode,
            'policy_state_dict': self.policy.state_dict(),
            'ppo_state_dict': self.ppo.policy_old.state_dict() if self.ppo else None,
            'metrics': self.metrics,
            'config': self.config
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Чекпоинт сохранён: {checkpoint_path}")
    
    def train(
        self, 
        save_filename: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ) -> torch.nn.Module:
        """
        Запустить процесс обучения.
        
        Args:
            save_filename: Путь для сохранения финальной модели
            checkpoint_path: Путь для загрузки/сохранения чекпоинтов
        
        Returns:
            Обученная политика
        """
        try:
            self._start_time = time.time()
            self._status = TrainingStatus.RUNNING
            self.metrics.status = TrainingStatus.RUNNING
            
            # Инициализация PPO
            self.ppo = PPO(self.policy)
            
            # Загрузка чекпоинта если указан
            start_episode = 0
            if checkpoint_path:
                try:
                    self.load_checkpoint(checkpoint_path)
                    start_episode = self.metrics.episode
                except FileNotFoundError:
                    print(f"Чекпоинт не найден, начинаем с начала")
            
            # Обновляем конфигурацию
            if save_filename:
                self.config.save_filename = save_filename
            
            # 1. Создаем коллектор и память ОДИН РАЗ
            collector = SharedMemoryExperienceCollector(
                self.config.num_workers, 
                self.config.max_steps_per_worker
            )
            state_size = Constants.state_size
            action_size = Constants.action_size
            
            shm_objects, total_size, step_size = SharedMemoryExperienceCollector.create_shared_memory_blocks(
                self.config.num_workers, 
                self.config.max_steps_per_worker, 
                state_size, 
                action_size
            )
            
            # Инициализация delay policy
            delay_policy = SeparateNetworkPolicy()
            old_state = delay_policy.state_dict()
            new_state = delay_policy.state_dict()
            
            # 2. Создаем пул процессов ОДИН РАЗ
            with mp.Pool(processes=self.config.num_workers) as pool:
                # 3. Основной цикл обучения
                for episode in range(start_episode, self.config.num_episodes):
                    # Проверка статуса
                    if self._status == TrainingStatus.PAUSED:
                        while self._status == TrainingStatus.PAUSED:
                            time.sleep(0.1)
                        if self._status != TrainingStatus.RUNNING:
                            break
                    
                    if self._status != TrainingStatus.RUNNING:
                        break
                    
                    # Проверка времени обучения
                    if self.config.max_training_time:
                        elapsed = time.time() - self._start_time
                        if elapsed > self.config.max_training_time:
                            self.stop("Превышено максимальное время обучения")
                            break
                    
                    self.metrics.episode = episode
                    episode_start = time.time()
                    
                    # Получаем текущие веса политики
                    state_dict1 = self.ppo.policy.cpu().state_dict()
                    
                    # Обновляем delay policy
                    if episode % self.config.delay_ep == 0:
                        old_state = new_state
                        new_state = state_dict1
                    
                    # 3.1. Запускаем worker'ов в существующем пуле
                    args = [
                        (self.config.max_steps_per_worker, step_size, shm_objects, state_dict1, old_state, i)
                        for i in range(self.config.num_workers)
                    ]
                    
                    try:
                        pool.starmap(SharedMemoryExperienceCollector.worker_shared_memory, args)
                    except Exception as e:
                        self._status = TrainingStatus.ERROR
                        self.metrics.error_message = f"Ошибка сбора опыта: {str(e)}"
                        print(f"ERROR: {self.metrics.error_message}")
                        break
                    
                    # 3.2. Собираем данные из памяти
                    try:
                        exp = SharedMemoryExperienceCollector.read_and_parse_shared_memory(
                            total_size, step_size, state_size, action_size, shm_objects
                        )
                    except Exception as e:
                        self._status = TrainingStatus.ERROR
                        self.metrics.error_message = f"Ошибка чтения памяти: {str(e)}"
                        print(f"ERROR: {self.metrics.error_message}")
                        break
                    
                    # 3.3. Обновление PPO
                    episode_reward = 0.0
                    if exp and len(exp.states) > 0:
                        try:
                            self.ppo.update_combined(exp, episode)
                            
                            # Считаем награду за эпизод
                            if hasattr(exp, 'rewards') and len(exp.rewards) > 0:
                                episode_reward = sum(exp.rewards) / len(exp.rewards)
                                self.metrics.rewards_history.append(episode_reward)
                                self.metrics.total_reward += episode_reward
                                self.metrics.avg_reward = self.metrics.total_reward / (episode - start_episode + 1)
                        except Exception as e:
                            self._status = TrainingStatus.ERROR
                            self.metrics.error_message = f"Ошибка обновления PPO: {str(e)}"
                            print(f"ERROR: {self.metrics.error_message}")
                            break
                    
                    # 3.4. Сохранение модели
                    if self.config.save_filename and episode % self.config.save_interval == 0:
                        torch.save(self.ppo.policy.state_dict(), self.config.save_filename)
                    
                    # 3.5. Сохранение чекпоинта
                    if self.config.checkpoint_dir and episode % self.config.checkpoint_interval == 0:
                        checkpoint_file = f"{self.config.checkpoint_dir}/checkpoint_ep{episode}.pt"
                        self.save_checkpoint(checkpoint_file)
                    
                    # 3.6. Очистка shared memory
                    for shm_name in shm_objects:
                        try:
                            if isinstance(shm_name, str):
                                buffer = ChunkedMmapBuffer()
                                buffer.close(shm_name, delete_files=False, clear_files=True)
                        except Exception as e:
                            print(f"Warning: ошибка очистки shared memory: {e}")
                    
                    # 3.7. Логирование
                    episode_time = time.time() - episode_start
                    if episode % self.config.log_interval == 0:
                        print(f"Episode {episode + 1}/{self.config.num_episodes} "
                              f"({(episode + 1) * 100 / self.config.num_episodes:.1f}%) "
                              f"| Reward: {episode_reward:.4f} "
                              f"| Avg: {self.metrics.avg_reward:.4f} "
                              f"| Time: {episode_time:.2f}s")
                    
                    # 3.8. Callback
                    if self._callback:
                        self.metrics.training_time = time.time() - self._start_time
                        self.metrics.episodes_completed = episode - start_episode + 1
                        if not self._callback(self.metrics):
                            self.stop("Остановлено callback'ом")
                            break
                    
                    # 3.9. Ранняя остановка
                    if self.config.early_stop_reward and self.metrics.avg_reward >= self.config.early_stop_reward:
                        print(f"Ранняя остановка: достигнута средняя награда {self.metrics.avg_reward:.4f}")
                        self._status = TrainingStatus.COMPLETED
                        break
                
                # Завершено успешно
                if self._status == TrainingStatus.RUNNING:
                    self._status = TrainingStatus.COMPLETED
                    self.metrics.status = TrainingStatus.COMPLETED
            
            # Финальное сохранение
            if self.config.save_filename and self._status == TrainingStatus.COMPLETED:
                torch.save(self.ppo.policy.state_dict(), self.config.save_filename)
                print(f"Модель сохранена: {self.config.save_filename}")
            
            # Финальные метрики
            self.metrics.training_time = time.time() - self._start_time
            self.metrics.episodes_completed = self.config.num_episodes - start_episode
            
            print(f"\nОбучение завершено: {self._status.value}")
            print(f"Всего эпизодов: {self.metrics.episodes_completed}")
            print(f"Средняя награда: {self.metrics.avg_reward:.4f}")
            print(f"Время обучения: {self.metrics.training_time:.2f}s")
            
            return self.policy
            
        except Exception as e:
            self._status = TrainingStatus.ERROR
            self.metrics.error_message = str(e)
            print(f"CRITICAL ERROR: {e}")
            raise
        
        finally:
            # Очистка ресурсов
            try:
                for shm_name in shm_objects:
                    if isinstance(shm_name, str):
                        buffer = ChunkedMmapBuffer()
                        buffer.close(shm_name, delete_files=True, clear_files=False)
            except:
                pass
    
    def train_simple(self, num_episodes: Optional[int] = None) -> torch.nn.Module:
        """
        Упрощённый метод для быстрого старта без сложной конфигурации.
        """
        if num_episodes:
            self.config.num_episodes = num_episodes
        
        return self.train()
