import torch

from AI.Training.Collector import SharedMemoryExperienceCollector
from AI.Training.Memory import Memory
from AI.Training.ChunkedMmapBuffer import ChunkedMmapBuffer
from AI.Training.PPO import PPO
from AI.Training.Environment import Environment
import Constants
from AI.SeparateNetworkPolicy import SeparateNetworkPolicy

import multiprocessing as mp
from multiprocessing import shared_memory


class Training_process:
    def __init__(self, env : Environment, draw_stats = False, num_episodes = 100):
        self.num_episodes = num_episodes
        self.batch_size = 4096 * 5
        self.memory = Memory()
        self.ppo = PPO(env.nn1)
        self.env = env
        self.save = None
        self.draw_stats = draw_stats
        self.logging = False
        

    def train(self, save_filename=None, num_workers=1, max_steps_per_worker=1024):        
        # 1. Создаем коллектор и память ОДИН РАЗ
        collector = SharedMemoryExperienceCollector(num_workers, max_steps_per_worker)
        state_size = Constants.state_size
        action_size = Constants.action_size
        # Этот метод должен только создавать shm, но не запускать сбор
        shm_objects, total_size, step_size = SharedMemoryExperienceCollector.create_shared_memory_blocks(num_workers, max_steps_per_worker, state_size, action_size) 
        
        delay_ep = 10
        delay_policy = SeparateNetworkPolicy()
        old_state = delay_policy.state_dict()
        new_state = delay_policy.state_dict()

        # 2. Создаем пул процессов ОДИН РАЗ
        with mp.Pool(processes=num_workers) as pool:
            # 3. Теперь запускаем основной цикл
            for episode in range(0, self.num_episodes):
                state_dict1 = self.ppo.policy.cpu().state_dict()

                if (episode) % delay_ep == 0:
                    old_state = new_state
                    new_state = state_dict1
                
                # --- ВАШ СТАРЫЙ `collect_experience_shared` нужно разделить ---
                
                # 3.1. Запускаем worker-ов в существующем пуле
                args = [
                    (max_steps_per_worker, step_size, shm_objects, state_dict1, state_dict1, i) 
                    for i in range(num_workers)
                ]
                pool.starmap(SharedMemoryExperienceCollector.worker_shared_memory, args) # Передаем метод worker-а

                # 3.2. Собираем данные из памяти (этот код можно вынести в отдельный метод)
                exp1 = SharedMemoryExperienceCollector.read_and_parse_shared_memory(total_size, step_size, state_size, action_size, shm_objects)
                
                # 3.3. Обновление PPO
                if exp1:
                    self.ppo.update_combined(exp1, episode)

                if save_filename and episode % 20 == 0: # Сохраняем реже
                    torch.save(self.ppo.policy.state_dict(), save_filename)

                for shm in shm_objects:
                    if isinstance(shm, ChunkedMmapBuffer):
                        shm.close(shm, delete_files=False)

                print(f"Episode {(episode+1) * 100 / self.num_episodes}%")

        # 4. Очищаем память ОДИН РАЗ в самом конце
        # for shm in collector.shm_objects:
        #     if isinstance(shm, ChunkedMmapBuffer):
        #         shm.close(delete_files=True)

        return self.ppo.policy
