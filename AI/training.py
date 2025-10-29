import numpy as np
import torch
from AI.memory import Memory
from AI.ppo import PPO
from AI.Enviroment import Enviroment
from Draw.Drawing import Drawing
import pygame

class Training:
    def __init__(self, env : Enviroment):
        self.num_episodes = 10000
        self.batch_size = 200
        self.memory = Memory()
        self.ppo = PPO(env.ai_action.translator.net)
        self.env = env
        self.last_rewards = -1000
        self.save = None
        

    def train(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            done = False
            
            d = Drawing(self.env.map)
            while not done:
                # Используем get_action для сэмплирования
                action, log_prob = self.ppo.select_action(state)
                
                # Выполняем действие в среде
                next_state, reward, done = self.env.step(action)
                d.draw()
                pygame.display.update()
                
                # Сохраняем в memory
                self.memory.store(state, action, log_prob, reward, done)
                
                state = next_state

            # if(np.array(self.memory.rewards).mean() > self.last_rewards):
            self.save = torch.save(self.ppo.policy.state_dict(), f"{self.ppo.policy.name}.pth")

            # Когда собрали достаточно данных - обучаем
            if len(self.memory.states) >= self.batch_size:
                self.ppo.update(self.memory, episode)
                self.memory.clear()

            print(f"Episode {(episode+1) * 100 / self.num_episodes}%")

        return self.ppo.policy