import copy
import numpy as np
import torch

from AI.Maksigma_net import Maksigma_net
from AI.TrajectoryDataset import TrajectoryDataset
from AI.memory import Memory
from AI.ppo import PPO
from AI.Enviroment import Enviroment
import Constants
from Draw.Drawing import Drawing
from Objects.Map import clone_and_detach_map

import pygame
import torch.multiprocessing as mp

class Training:
    def __init__(self, env : Enviroment, train_loader, draw = False):
        self.num_episodes = 100
        self.batch_size = 4096 * 5
        self.memory = Memory()
        self.ppo = PPO(env.ai_action.translator.net)
        self.env = env
        self.last_rewards = -1000
        self.save = None
        self.draw = draw
        self.train_loader = train_loader
        self.logging = False
        

    def train(self, logging = True):
        self.logging = logging

        merged_list = []

        for episode in range(self.num_episodes):
            
            state_dict = self.ppo.policy.state_dict() # копии нейронки
            env_map = [clone_and_detach_map(self.env.map) for _ in range(20)] # создание копии карты для обучения

            # args = [(env_map[i], state_dict) for i in range(10)] # 
            # # подготовьте разные карты: [(env_map1, ...), (env_map2, ...), ...]

            # with mp.Pool(processes=10) as pool:
            #     results = pool.starmap(worker, args)
            state_dict = self.ppo.policy.cpu().state_dict()
        
            processes = []
            for i in range(20):
              p = mp.Process(target=worker_torch, args=(i, env_map[i], state_dict))
              p.start()
              processes.append(p)
        
            # Сбор результатов
            results = []
            for p in processes:
              p.join()
              # Получить результат через shared memory или queue
              results.append(p.exitcode)  # Или через Queue
        
            for p in processes:
              p.close()
            
            # Слияние results = trajectories

            # state = self.env.reset()
            # done = False
            
            # d = Drawing(self.env.map)
            # while not done:
            #     # Используем get_action для сэмплирования
            #     action, log_prob = self.ppo.select_action(state)
                
            #     # Выполняем действие в среде
            #     next_state, reward, done = self.env.step(action)
            #     if self.draw:
            #         d.draw()
            #         pygame.display.update()
                
            #     # Сохраняем в memory
            #     self.memory.store(state, action, log_prob, reward, done)
                
            #     state = next_state

            # if(episode % 100 == 0):
            #     self.save = torch.save(self.ppo.policy.state_dict(), f"{self.ppo.policy.name}.pth")
            



            # states = torch.stack([s.detach() for s in self.memory.states])
            # actions = torch.stack([s.detach() for s in self.memory.actions])
            # # old_log_probs = torch.tensor(, device=Constants.device)  # От policy_old
            # old_log_probs = torch.stack([s.detach() for s in self.memory.old_log_probs])  # От policy_old
            # rewards = torch.stack([s.detach() for s in self.memory.rewards])  # От policy_old
            # rewards = 

            
            merged_list.append(merge_memories(results))
            merged = merge_memories(merged_list)
            # Когда собрали достаточно данных - обучаем
            if len(merged.actions) >= self.batch_size:

                
                states = torch.stack(merged.states).to(Constants.device)        # shape: (total_steps, state_size)
                actions = torch.stack(merged.actions).to(Constants.device)        
                rewards = torch.tensor(merged.rewards).to(Constants.device)        
                log_probs = torch.tensor(merged.old_log_probs).to(Constants.device)        

                # dataset = TrajectoryDataset(states, actions, rewards, old_log_probs)
                # self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

                # for batch in self.train_loader:
                #     batch_states = batch['state'].to(Constants.device, non_blocking=True)
                #     batch_actions = batch['action'].to(Constants.device, non_blocking=True)
                #     batch_rewards = batch['reward'].to(Constants.device, non_blocking=True)
                #     batch_log_probs = batch['log_prob'].to(Constants.device, non_blocking=True)

                self.ppo.update(batch_states=states, batch_actions=actions,
                                     batch_rewards=rewards, batch_logps=log_probs, ep=episode, logging=self.logging)
                    
                    
                torch.save(self.ppo.policy.state_dict(), f"{self.ppo.policy.name}.pth")
                self.memory.clear()
                merged_list.clear()

            print(f"Episode {(episode+1) * 100 / self.num_episodes}%")

        return self.ppo.policy

def collect_trajectory(env, policy):
    memory = Memory()
    state = env.reset()
    done = False
    while not done:
    #     # Используем get_action для сэмплирования
        action, log_prob = policy.select_action(state)
                
                # Выполняем действие в среде
        next_state, reward, done = env.step(action)
                
                # Сохраняем в memory
        memory.store(state, action, log_prob, reward, done)
                
        state = next_state
    return memory

def worker(env_map, state_dict):
    # 1. Инициализация своей среды, policy
    # env_map = clone_and_detach_map(env_map)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = Constants.device

    nn = Maksigma_net()
    nn.load_state_dict(state_dict)
    nn.to(device)
    env = Enviroment(env_map, nn)
    
    # 2. Сбор trajectory
    traj = collect_trajectory(env, nn)
    return traj

def worker_torch(rank, env_map, state_dict):
    """Worker для torch.multiprocessing"""
    torch.cuda.set_device(rank % torch.cuda.device_count())
    
    local_policy = Maksigma_net()
    local_policy.load_state_dict(state_dict)
    local_policy = local_policy.to(f'cuda:{rank % torch.cuda.device_count()}')
    env = Enviroment(env_map, local_policy)
    
    experiences = collect_trajectory(env, local_policy)
    
    # Важно: перенести на CPU
    experiences.to_device(Constants.device)
    return experiences

def merge_memories(memories):
    merged = Memory()
    for mem in memories:
        merged.states.extend(mem.states)
        merged.actions.extend(mem.actions)
        merged.old_log_probs.extend(mem.old_log_probs)
        merged.rewards.extend(mem.rewards)
        merged.is_terminals.extend(mem.is_terminals)
    return merged