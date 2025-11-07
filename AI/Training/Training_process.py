import torch

from AI.Training.Collector import SharedMemoryExperienceCollector
from AI.Training.Memory import Memory
from AI.Training.PPO import PPO
from AI.Training.Environment import Environment
import Constants

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
        

    def train(self, num_workers = 10, max_steps_per_worker = 1024, draw_stats = True, \
              save_filename = None):
        self.logging = draw_stats

        self.env.num_steps = max_steps_per_worker

        collector = SharedMemoryExperienceCollector(num_workers, max_steps_per_worker)

        for episode in range(self.num_episodes):
            state_dict1 = self.ppo.policy.cpu().state_dict()  # CPU для безопасной передачи
            
            exp1 = collector.collect_experience_shared(state_dict1, Constants.state_size, Constants.action_size)
            self.shm_objects1 = collector.shm_objects

            # Обновление PPO
            if exp1:
                self.ppo.update_with_minibatches_default(exp1, episode)
                # self.ppo.update_with_minibatches(exp1, episode)
                # self.ppo.update(exp1, episode, self.logging)

            for shm in collector.shm_objects:
                shm.close()
                shm.unlink()

            if save_filename is not None:
                self.save1 = torch.save(self.ppo.policy.state_dict(), save_filename)

            print(f"Episode {(episode+1) * 100 / self.num_episodes}%")

        return self.ppo.policy