import multiprocessing as mp
import numpy as np
from multiprocessing import shared_memory
import torch

from AI.Maksigma_net import Maksigma_net
from AI.SeparateNetworkPolicy import SeparateNetworkPolicy
from AI.Simple import UltraSimplePolicy
from AI.Training.Environment import Environment
from AI.Training.Memory import Memory
from AI.Training.ChunkedMmapBuffer import ChunkedMmapBuffer
import Constants

class SharedMemoryExperienceCollector:
    def __init__(self, num_workers=20, max_steps_per_worker=1000):
        self.num_workers = num_workers
        self.max_steps_per_worker = max_steps_per_worker
        self.step_size = None

    def create_or_cleanup_shared_memory(self, name, size):
        try:
            # Попытка открыть существующий сегмент
            existing_shm = shared_memory.SharedMemory(name=name)
            existing_shm.close()
            existing_shm.unlink()
            print(f"Старый сегмент {name} удалён")
        except FileNotFoundError:
            pass  # Нет старого сегмента — всё ок

        # Создаём новый сегмент
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        
        return shm
        
    def create_shared_memory_blocks(num_workers, max_steps_per_worker, state_size, action_size):
        """Создание shared memory блоков для каждого worker'а"""
        shm_objects = []
        
        for i in range(num_workers):
            # размер одной ячейки сохранения в байтах
            step_bytes = (state_size + action_size + 1 + 1 + 1 + 1) * 4 * 2  # *2 — для двух агентов
            step_size = state_size + action_size + 1 + 1 + 1 + 1  # в элементах float32, без умножений
            total_size = max_steps_per_worker * step_bytes
            
            shm_name = f"ppo_experience_{i}.dat"

            # shm = self.create_or_cleanup_shared_memory(shm_name, self.total_size)
            shm = ChunkedMmapBuffer()
            shm.create(shm_name, max_steps_per_worker * 2, step_size)

            # shm = shared_memory.SharedMemory(create=True, size=self.total_size, name=shm_name)
            # self.shm_objects.append(shm)  # Сохраняем, чтобы не удалил сборщик мусора
            # self.shm_names.append(shm_name)
            shm_objects.append(shm_name)

            # self.shapes.append((self.max_steps_per_worker * 2, action_size, state_size))  # Flexible shape
             
        
        return shm_objects, total_size, step_size
    
    @staticmethod
    def worker_shared_memory(max_steps_per_worker, step_size, shm_names, state_dict1, state_dict2, process_id):
        """Worker: собирает опыт и пишет в shared memory"""
        try:
            # Загрузка модели (на CPU сначала)
            local_policy = SeparateNetworkPolicy()  # Ваш класс политики
            local_policy_old = SeparateNetworkPolicy()  # Ваш класс политики
            local_policy.load_state_dict(state_dict1)
            local_policy_old.load_state_dict(state_dict2)
            
            # Инициализация shared memory
            shm_name = shm_names[process_id]
            
            shm = ChunkedMmapBuffer()
            shm.open(max_steps_per_worker * 2, step_size, shm_name)
            experience_buffer = shm
                
            
            # Сбор опыта
            step = 0
            env = Environment(local_policy, local_policy_old, max_steps_per_worker)
            
            done = False
            s1, s2 = env.reset()
            while not done:  # Ваш цикл игры

                action1, log_prob1 = local_policy.select_action(s1)
                action2, log_prob2 = local_policy_old.select_action(s2)
                
                # action1 = torch.tensor([0.1, 0, 0])
                # action2 = torch.tensor([0.1, 0, 0])

                # Готовим действия для шага в среде
                action_to_apply_1 = action1.detach().clone()
                action_to_apply_2 = action2.detach().clone()

                action_to_apply_2[0] = -action_to_apply_2[0]

                (ns1, ns2), (r1, r2), done, info = env.step(action_to_apply_1, action_to_apply_2)
                # print(f"action1={action1} state1={s1} reward1={r1} done={done}")
                # print(f"action2={action2} state2={s2} reward2={r2} done={done}")
        
                # Извлекаем флаги
                natural_done = info.get('natural_done', False)
                truncated = info.get('truncated', False)

                if natural_done or truncated:
                    pass
                
                test1 = s1.detach().flatten().numpy()
                test2 = action1.detach().flatten().numpy()
                test3 = np.array([r1.flatten().detach().numpy().item()])
                test4 = np.array([log_prob1.flatten().detach().numpy().item()])
                test5 = np.array([1.0 if natural_done else 0.0])
                test6 = np.array([1.0 if truncated else 0.0])
                # Сохранить опыт (flattened)
                experience_line1 = np.concatenate([
                    test1,
                    test2,  
                    test3,  
                    test4, 
                    test5,  
                    test6   
                ])
                experience_buffer[step] = experience_line1  # Сохраняем в текущую ячейку

                # --- Опыт для Игрока 2 ---
                exp2_state = s2.detach().flatten().numpy()
                exp2_action = action2.detach().flatten().numpy()
                exp2_reward = np.array([r2.item()])
                exp2_log_prob = np.array([log_prob2.item()])
                exp2_done = np.array([1.0 if natural_done else 0.0])
                exp2_truncated = np.array([1.0 if truncated else 0.0])

                experience_line2 = np.concatenate([exp2_state, exp2_action, exp2_reward, exp2_log_prob, exp2_done, exp2_truncated])
                # experience_buffer[step + 1] = experience_line2 # Сохраняем в следующую ячейку

                # print(f"exp_line1={experience_line1}")
                # print(f"exp_line2={experience_line2}")

                step += 2 # Увеличиваем шаг на 2

                s1, s2 = ns1, ns2

            shm.close(shm_name, delete_files=False, clear_files=False)
            return process_id  # Успешное завершение
            
        except Exception as e:
            print(f"Worker {process_id} error: {e}")
            return None
    
    def collect_experience_shared(self, state_dict1, state_dict2, state_size, action_size):
        """Основная функция: запуск worker'ов и сбор результатов"""
        if __name__ == '__main__':
            mp.set_start_method('spawn')
        
        # Создание shared memory
        shm_names, shapes = self.create_shared_memory_blocks(state_size, action_size)
        
        # Аргументы для worker'ов
        args = [
            (self.step_size, state_dict1, state_dict2, shm_names[i], i) 
            for i in range(self.num_workers)
        ]
        
        # Запуск worker'ов
        with mp.Pool(processes=self.num_workers) as pool:
            pool.starmap(self.worker_shared_memory, args)
        
        # Сбор данных из shared memory
        all_exp_states1, all_exp_actions1, all_exp_log_probs1, all_exp_rewards1, all_exp_dones1, all_exp_truncated = [], [], [], [], [], []
        
        for i, _ in enumerate(shm_names):
            try:
                # Открытие shared memory
                shm = self.shm_objects[i]
                
                # Чтение данных
                num_steps = self.total_size // (self.step_size * 4)
                
                if isinstance(shm, ChunkedMmapBuffer):
                    shm.open()
                    experience_buffer = shm
                
                # Парсинг опыта
                
                for step in range(num_steps):
                    if np.all(experience_buffer[step] == 0): continue 

                    line = experience_buffer[step]
                    state = line[0:state_size]
                    action = line[state_size:state_size + action_size]
                    reward = line[state_size + action_size]
                    log_prob = line[state_size + action_size + 1]
                    done = line[state_size + action_size + 2]
                    truncated = line[state_size + action_size + 3]


                    all_exp_states1.append(state.tolist())      
                    all_exp_actions1.append(action.tolist())
                    all_exp_log_probs1.append(float(log_prob))
                    all_exp_rewards1.append(float(reward))
                    all_exp_dones1.append(float(done))
                    all_exp_truncated.append(float(truncated))
                
            except Exception as e:
                print(f"Error reading shared memory {i}: {e}")
                continue

        # После чтения shared memory из всех процессов
        merged1 = Memory()
        merged1.states = all_exp_states1
        merged1.actions_final = all_exp_actions1
        merged1.old_log_probs = all_exp_log_probs1
        merged1.rewards = all_exp_rewards1
        merged1.is_terminals = all_exp_dones1
        merged1.is_truncated = all_exp_truncated
        merged1.copy_to_tensors()

        return merged1
    
    def read_and_parse_shared_memory(total_size, step_size,  state_size, action_size, shm_names):
        # Сбор данных из shared memory
        all_exp_states1, all_exp_actions1, all_exp_log_probs1, all_exp_rewards1, all_exp_dones1, all_exp_truncated = [], [], [], [], [], []
        

        shm_objects = []
        for i in range(len(shm_names)):
            try:
                # Открытие shared memory
                shm_name = shm_names[i]
                shm = ChunkedMmapBuffer()
                
                # Чтение данных
                num_steps = total_size // (step_size * 4)
                shm.open(num_steps, step_size, shm_name)
                
                if isinstance(shm, ChunkedMmapBuffer):
                    # shm.open()
                    experience_buffer = shm
                    shm_objects.append(shm)
                
                # Парсинг опыта
                
                for step in range(num_steps):
                    if np.all(experience_buffer[step] == 0): 
                        continue 

                    line = experience_buffer[step]
                    state = line[0:state_size]
                    action = line[state_size:state_size + action_size]
                    reward = line[state_size + action_size]
                    log_prob = line[state_size + action_size + 1]
                    done = line[state_size + action_size + 2]
                    truncated = line[state_size + action_size + 3]


                    all_exp_states1.append(state.tolist())      
                    all_exp_actions1.append(action.tolist())
                    all_exp_log_probs1.append(float(log_prob))
                    all_exp_rewards1.append(float(reward))
                    all_exp_dones1.append(float(done))
                    all_exp_truncated.append(float(truncated))
                
            except Exception as e:
                print(f"Error reading shared memory {i}: {e}")
                continue

        # После чтения shared memory из всех процессов
        merged1 = Memory()
        merged1.states = all_exp_states1
        merged1.actions_final = all_exp_actions1
        merged1.old_log_probs = all_exp_log_probs1
        merged1.rewards = all_exp_rewards1
        merged1.is_terminals = all_exp_dones1
        merged1.is_truncated = all_exp_truncated
        merged1.copy_to_tensors()

        for i in range(len(shm_objects)):
            shm_name = shm_names[i]
            shm = shm_objects[i]
            if isinstance(shm, ChunkedMmapBuffer):
                shm.close(shm_name, delete_files=False)

        return merged1
