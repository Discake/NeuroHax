import numpy as np
import torch

from AI.TrajectoryDataset import TrajectoryDataset
from AI.memory import Memory
from AI.ppo import PPO
from AI.Enviroment import Enviroment
import Constants
from Draw.Drawing import Drawing

import pygame

class Training:
    def __init__(self, env : Enviroment, train_loader, draw = False):
        self.num_episodes = 10000
        self.batch_size = 256
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
        for episode in range(self.num_episodes):
            state = self.env.reset()
            done = False
            
            d = Drawing(self.env.map)
            while not done:
                # Используем get_action для сэмплирования
                action, log_prob = self.ppo.select_action(state)
                
                # Выполняем действие в среде
                next_state, reward, done = self.env.step(action)
                if self.draw:
                    d.draw()
                    pygame.display.update()
                
                # Сохраняем в memory
                self.memory.store(state, action, log_prob, reward, done)
                
                state = next_state

            if(episode % 100 == 0):
                self.save = torch.save(self.ppo.policy.state_dict(), f"{self.ppo.policy.name}.pth")


            states = torch.stack([s.detach() for s in self.memory.states])
            actions = torch.stack([s.detach() for s in self.memory.actions])
            # old_log_probs = torch.tensor(, device=Constants.device)  # От policy_old
            old_log_probs = torch.stack([s.detach() for s in self.memory.old_log_probs])  # От policy_old
            rewards = torch.stack([s.detach() for s in self.memory.rewards])  # От policy_old
            # rewards = 

            

            # Когда собрали достаточно данных - обучаем
            if len(self.memory.states) >= self.batch_size:

                dataset = TrajectoryDataset(states, actions, rewards, old_log_probs)
                self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

                for batch in self.train_loader:
                    batch_states = batch['state'].to(Constants.device, non_blocking=True)
                    batch_actions = batch['action'].to(Constants.device, non_blocking=True)
                    batch_rewards = batch['reward'].to(Constants.device, non_blocking=True)
                    batch_log_probs = batch['log_prob'].to(Constants.device, non_blocking=True)

                    self.ppo.update(batch_states=batch_states, batch_actions=batch_actions,
                                     batch_rewards=batch_rewards, batch_logps=batch_log_probs, ep=episode, logging=self.logging)
                self.memory.clear()

            print(f"Episode {(episode+1) * 100 / self.num_episodes}%")

        return self.ppo.policy