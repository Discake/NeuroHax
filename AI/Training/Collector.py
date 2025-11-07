import multiprocessing as mp
import numpy as np
from multiprocessing import shared_memory
import torch

from AI.Maksigma_net import Maksigma_net
from AI.Simple import UltraSimplePolicy
from AI.Training.Environment import Environment
from AI.Training.Memory import Memory
import Constants

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
            # step_size = state_size * 4 + action_size * 2 * 4 + 1 * 4 + 1 * 4 + 1 * 4 # bytes

            step_size *= 2 # Из-за двух нейросетей

            # размер одной ячейки сохранения в абсолютных единицах
            self.step_size = state_size + action_size + 1 + 1 + 1 # 3 единицы - награда, логарифм, шаг завершения
            # self.step_size = state_size + action_size * 2 + 1 + 1 + 1 # 3 единицы - награда, логарифм, шаг завершения

            self.total_size = self.max_steps_per_worker * step_size
            
            shm_name = f"ppo_experience_{i}"
            shm = shared_memory.SharedMemory(create=True, size=self.total_size, name=shm_name)
            self.shm_objects.append(shm)  # Сохраняем, чтобы не удалил сборщик мусора
            self.shm_names.append(shm_name)
            
            self.shapes.append((self.max_steps_per_worker, action_size, state_size))  # Flexible shape
             
        
        return self.shm_names, self.shapes
    
    def worker_shared_memory(self, step_size, state_dict, shm_name, process_id):
        """Worker: собирает опыт и пишет в shared memory"""
        try:
            # Загрузка модели (на CPU сначала)
            local_policy = UltraSimplePolicy()  # Ваш класс политики
            local_policy.load_state_dict(state_dict)
            
            # if torch.cuda.is_available():
            #     local_policy1 = local_policy1.to(f'cuda:{process_id % torch.cuda.device_count()}')
            #     local_policy2 = local_policy2.to(f'cuda:{process_id % torch.cuda.device_count()}')
            
            # Инициализация shared memory
            shm = self.shm_objects[process_id]
            
            # Создание numpy array для записи
            # Shape: (max_steps, num_features) - flattened
            buffer_size = shm.size
            num_steps = buffer_size // (step_size * 4)
            
            experience_buffer = np.ndarray((num_steps, step_size), dtype=np.float32, buffer=shm.buf)
            
            # Сбор опыта
            step = 0
            env = Environment(local_policy, local_policy, self.max_steps_per_worker)
            
            done = False
            s1, s2 = env.reset()
            while step < self.max_steps_per_worker and not done:  # Ваш цикл игры                
                if torch.cuda.is_available():
                    s1 = s1.to(Constants.device)
                    s2 = s2.to(Constants.device)

                action1, log_prob1 = local_policy.select_action(s1)
                action2, log_prob2 = local_policy.select_action(s2)
                # action1, action1_raw, log_prob1 = local_policy.select_action(s1)
                # action2, action2_raw, log_prob2 = local_policy.select_action(s2)
                (ns1, ns2), (r1, r2), done = env.step(action1, action2)
                
                test1 = s1.detach().flatten().numpy()
                test2 = action1.detach().flatten().numpy()
                # test2_raw = action1_raw.detach().flatten().numpy()
                test3 = np.array([r1.flatten().detach().numpy().item()])
                test4 = np.array([log_prob1.flatten().detach().numpy().item()])
                test5 = np.array([1.0 if done else 0.0])
                # Сохранить опыт (flattened)
                experience_line1 = np.concatenate([
                    test1,  # 8 floats
                    test2,  # 3 floats
                    # test2_raw,  # 3 floats
                    test3,  # 1 float
                    test4,  # 1 floats
                    test5   # 1 float
                ])
                test1 = s2.detach().flatten().numpy()
                test2 = action2.detach().flatten().numpy()
                # test2_raw = action2_raw.detach().flatten().numpy()
                test3 = np.array([r2.flatten().detach().numpy().item()])
                test4 = np.array([log_prob2.flatten().detach().numpy().item()])
                test5 = np.array([1.0 if done else 0.0])
                # Сохранить опыт (flattened)
                experience_line2 = np.concatenate([
                    test1,
                    test2,
                    # test2_raw,
                    test3,
                    test4,
                    test5
                ])
                

                if step % 100 == 0:
                    print(f"Записано шагов: {step}")
                
                experience_buffer[2 * step] = experience_line1
                experience_buffer[2 * step + 1] = experience_line2
                step += 1

                s1, s2 = ns1, ns2
                
            return process_id  # Успешное завершение
            
        except Exception as e:
            print(f"Worker {process_id} error: {e}")
            return None
    
    def collect_experience_shared(self, state_dict, state_size, action_size):
        """Основная функция: запуск worker'ов и сбор результатов"""
        if __name__ == '__main__':
            mp.set_start_method('spawn')
        
        # Создание shared memory
        shm_names, shapes = self.create_shared_memory_blocks(state_size, action_size)
        
        # Аргументы для worker'ов
        args = [
            (self.step_size, state_dict, shm_names[i], i) 
            for i in range(self.num_workers)
        ]
        
        # Запуск worker'ов
        with mp.Pool(processes=self.num_workers) as pool:
            pool.starmap(self.worker_shared_memory, args)
        
        # Сбор данных из shared memory
        all_exp_states1, all_exp_actions1, all_exp_actions1_raw, all_exp_log_probs1, all_exp_rewards1, all_exp_dones1 = [], [], [], [], [], []
        for i, _ in enumerate(shm_names):
            try:
                # Открытие shared memory
                shm = self.shm_objects[i]
                
                # Чтение данных
                num_steps = shm.size // (self.step_size * 4)
                experience_buffer = np.ndarray((num_steps, self.step_size), dtype=np.float32, buffer=shm.buf)
                
                # Парсинг опыта
                
                for step in range(num_steps * 2):
                    if np.all(experience_buffer[step] == 0): break  

                    # line = experience_buffer[step]
                    # state = line[0:state_size]
                    # action = line[state_size:state_size + action_size]
                    # action_raw = line[state_size+ action_size:state_size + action_size * 2]
                    # reward = line[state_size + action_size * 2]
                    # log_prob = line[state_size + action_size * 2 + 1]
                    # done = line[state_size + action_size * 2 + 2]
                    # all_exp_states1.append(state.tolist())      
                    # all_exp_actions1.append(action.tolist())
                    # all_exp_actions1_raw.append(action_raw.tolist())
                    # all_exp_log_probs1.append(float(log_prob))
                    # all_exp_rewards1.append(float(reward))
                    # all_exp_dones1.append(float(done))

                    line = experience_buffer[step]
                    state = line[0:state_size]
                    action = line[state_size:state_size + action_size]
                    # action_raw = line[state_size+ action_size:state_size + action_size * 2]
                    reward = line[state_size + action_size]
                    log_prob = line[state_size + action_size + 1]
                    done = line[state_size + action_size + 2]
                    all_exp_states1.append(state.tolist())      
                    all_exp_actions1.append(action.tolist())
                    # all_exp_actions1_raw.append(action_raw.tolist())
                    all_exp_log_probs1.append(float(log_prob))
                    all_exp_rewards1.append(float(reward))
                    all_exp_dones1.append(float(done))
                    # if done.item() > 0.5:
                    #     break

                    
                
                
            except Exception as e:
                print(f"Error reading shared memory {i}: {e}")
                continue

        # После чтения shared memory из всех процессов
        merged1 = Memory()
        merged1.states = all_exp_states1
        merged1.actions_final = all_exp_actions1
        # merged1.actions_raw = all_exp_actions1_raw
        merged1.old_log_probs = all_exp_log_probs1
        merged1.rewards = all_exp_rewards1
        merged1.is_terminals = all_exp_dones1
        merged1.copy_to_tensors()

        # merged2 = Memory()
        # merged2.states = all_exp_states2
        # merged2.actions = all_exp_actions2
        # merged2.old_log_probs = all_exp_log_probs2
        # merged2.rewards = all_exp_rewards2
        # merged2.is_terminals = all_exp_dones2
        # merged2.copy_to_tensors()
        
        return merged1
