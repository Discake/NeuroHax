import torch

from AI.collector import SharedMemoryExperienceCollector
from AI.memory import Memory
from AI.ppo import PPO
from AI.Environment import Environment
from Objects.Map import clone_and_detach_map

class Training:
    def __init__(self, env : Environment, draw_stats = False):
        self.num_episodes = 10000
        self.batch_size = 4096 * 5
        self.memory = Memory()
        self.ppo = PPO(env.ai_action.translator.net)
        self.env = env
        self.last_rewards = -1000
        self.save = None
        self.draw = draw_stats
        self.logging = False
        

    def train(self, logging = True):
        self.logging = logging

        collector = SharedMemoryExperienceCollector(num_workers=10, max_steps_per_worker=self.env.num_steps)

        for episode in range(self.num_episodes):
            state_dict = self.ppo.policy.cpu().state_dict()  # CPU для безопасной передачи
            env_maps = [clone_and_detach_map(self.env.map) for _ in range(collector.num_workers)]
            
            state_size = 8
            experiences = collector.collect_experience_shared(state_dict, env_maps, state_size=state_size, action_size=3)
            self.shm_objects1 = collector.shm_objects

            
            # Обновление PPO
            if experiences:
                self.ppo.update(experiences, episode, self.logging)

            for shm in collector.shm_objects:
                shm.close()
                shm.unlink()

            self.save = torch.save(self.ppo.policy.state_dict(), f"{self.ppo.policy.name}_new_method.pth")

            print(f"Episode {(episode+1) * 100 / self.num_episodes}%")

        return self.ppo.policy

def merge_memories(memories):
    merged = Memory()
    for mem in memories:
        merged.states.extend(mem.states)
        merged.actions.extend(mem.actions)
        merged.old_log_probs.extend(mem.old_log_probs)
        merged.rewards.extend(mem.rewards)
        merged.is_terminals.extend(mem.is_terminals)
    return merged