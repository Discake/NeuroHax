import multiprocessing as mp
import numpy as np
from multiprocessing import shared_memory
import torch

from AI.Maksigma_net import Maksigma_net
from AI.Translator import Translator
from AI.Environment import Environment
from AI.memory import Memory
import Constants
from Objects.Map import Map

class SharedMemoryExperienceCollector:
    def __init__(self, num_workers=20, max_steps_per_worker=1000):
        self.num_workers = num_workers
        self.max_steps_per_worker = max_steps_per_worker
        self.step_size = None
        
    def create_shared_memory_blocks(self, state_size, action_size):
        """Создание shared memory блоков для каждого worker'а"""
        self.shm_names = []
        self.shapes = []
        self.shm_objects = []
        
        for i in range(self.num_workers):
            # размер одной ячейки сохранения в байтах
            step_size = state_size * 4 + action_size * 4 + 1 * 4 + 1 * 4 + 1 * 4 # bytes

            # размер одной ячейки сохранения в абсолютных единицах
            self.step_size = state_size + action_size + 1 + 1 + 1 # 3 единицы - награда, логарифм, шаг завершения

            total_size = self.max_steps_per_worker * step_size
            
            shm_name = f"ppo_experience_kaggle_{i}"
            shm = shared_memory.SharedMemory(create=True, size=total_size, name=shm_name)
            self.shm_objects.append(shm)  # Сохраняем, чтобы не удалил сборщик мусора
            self.shm_names.append(shm_name)
            
            self.shapes.append((self.max_steps_per_worker, action_size, state_size))  # Flexible shape
             
        
        return self.shm_names, self.shapes
    
    def worker_shared_memory(self, step_size, env_map : Map, state_dict, shm_name, process_id):
        """Worker: собирает опыт и пишет в shared memory"""
        try:
            # Загрузка модели (на CPU сначала)
            local_policy = Maksigma_net()  # Ваш класс политики
            local_policy.load_state_dict(state_dict)
            
            if torch.cuda.is_available():
                local_policy = local_policy.to(f'cuda:{process_id % torch.cuda.device_count()}')
            
            # Инициализация shared memory
            shm = self.shm_objects[process_id]
            
            # Создание numpy array для записи
            # Shape: (max_steps, num_features) - flattened
            buffer_size = shm.size
            num_steps = buffer_size // (step_size * 4)
            
            experience_buffer = np.ndarray((num_steps, step_size), dtype=np.float32, buffer=shm.buf)
            
            # Сбор опыта
            experiences = []
            step = 0
            env = Environment(env_map, local_policy)
            
            done = False
            state = env.reset()
            while step < self.max_steps_per_worker and not done:  # Ваш цикл игры                
                if torch.cuda.is_available():
                    state = state.to(Constants.device)
                
                action, log_prob = local_policy.select_action(state)
                next_state, reward, done = env.step(action)
                
                test1 = state.detach().flatten().cpu().numpy()


                test2 = action.detach().flatten().cpu().numpy()
                test3 = np.array([reward.flatten().cpu().detach().numpy().item()])
                test4 = np.array([log_prob.flatten().cpu().detach().numpy().item()])
                test5 = np.array([1.0 if done else 0.0])
                # Сохранить опыт (flattened)
                experience_line = np.concatenate([
                    test1,  # 8 floats
                    test2,  # 3 floats
                    test3,  # 1 float
                    test4,  # 1 floats
                    test5   # 1 float
                ])
                

                if step % 100 == 0:
                    print(f"Записано шагов: {step}")

                # Padding если нужно
                if len(experience_line) < step_size:
                    experience_line = np.pad(experience_line, (0, step_size - len(experience_line)))
                
                experience_buffer[step] = experience_line
                step += 1

                state = next_state

            # Заполнить оставшиеся строки нулями
            if step < num_steps:
                experience_buffer[step:] = 0
                
            # shm.close()
            return process_id  # Успешное завершение
            
        except Exception as e:
            print(f"Worker {process_id} error: {e}")
            return None
    
    def collect_experience_shared(self, state_dict, env_maps, state_size, action_size):
        """Основная функция: запуск worker'ов и сбор результатов"""
        if __name__ == '__main__':
            mp.set_start_method('spawn')

        memories = []
        
        # Создание shared memory
        shm_names, shapes = self.create_shared_memory_blocks(state_size, action_size)
        
        # Аргументы для worker'ов
        args = [
            (self.step_size, env_maps[i], state_dict, shm_names[i], i) 
            for i in range(self.num_workers)
        ]
        
        # Запуск worker'ов
        with mp.Pool(processes=self.num_workers) as pool:
            pool.starmap(self.worker_shared_memory, args)
        
        # Сбор данных из shared memory
        all_exp_states, all_exp_actions, all_exp_log_probs, all_exp_rewards, all_exp_dones = [], [], [], [], []
        for i, _ in enumerate(shm_names):
            try:
                # Открытие shared memory
                shm = self.shm_objects[i]
                
                # Чтение данных
                num_steps = shm.size // (self.step_size * 4)
                experience_buffer = np.ndarray((num_steps, self.step_size), dtype=np.float32, buffer=shm.buf)
                
                # Парсинг опыта
                
                for step in range(num_steps):
                    if np.all(experience_buffer[step] == 0): break    
                    line = experience_buffer[step]
                    state = line[0:state_size]
                    action = line[state_size:state_size + action_size]
                    reward = line[state_size + action_size]
                    log_prob = line[state_size + action_size + 1]
                    done = line[state_size + action_size + 2]
                    all_exp_states.append(state.tolist())      
                    all_exp_actions.append(action.tolist())
                    all_exp_log_probs.append(float(log_prob))
                    all_exp_rewards.append(float(reward))
                    all_exp_dones.append(float(done))
                
                
            except Exception as e:
                print(f"Error reading shared memory {i}: {e}")
                continue

        # После чтения shared memory из всех процессов
        merged = Memory()
        merged.states = all_exp_states
        merged.actions = all_exp_actions
        merged.old_log_probs = all_exp_log_probs
        merged.rewards = all_exp_rewards
        merged.is_terminals = all_exp_dones
        merged.copy_to_tensors()
        
        return merged

def merge_memories(memories):
    merged = Memory()
    for mem in memories:
        merged.states.extend(list(mem.states))
        merged.actions.extend(list(mem.actions))
        merged.old_log_probs.extend(list(mem.old_log_probs))
        merged.rewards.extend(list(mem.rewards))
    return merged 
