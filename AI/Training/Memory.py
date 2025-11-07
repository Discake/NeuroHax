import Constants
import torch

class Memory:
    def __init__(self):
        self.states = []
        self.actions_final = []      # Декартовы координаты velocity
        self.actions_raw = []        # Сырые angle, speed для log_prob
        self.old_log_probs = []
        self.rewards = []
        self.is_terminals = []
    
    def store(self, state, action_final, action_raw, log_prob, reward, done):
        self.states.append(state.tolist())
        self.actions_final.append(action_final.tolist())
        self.actions_raw.append(action_raw.tolist())     # НОВОЕ!
        self.old_log_probs.append(log_prob.item())
        self.rewards.append(reward.item())
        self.is_terminals.append(done)
    

    def copy_to_tensors(self):
        self.states = torch.tensor(self.states, device=Constants.device)
        self.actions_final = torch.tensor(self.actions_final, device=Constants.device)
        self.actions_raw = torch.tensor(self.actions_raw, device=Constants.device)
        self.old_log_probs = torch.tensor(self.old_log_probs, device=Constants.device)
        self.rewards = torch.tensor(self.rewards, device=Constants.device)
        self.is_terminals = torch.tensor(self.is_terminals, device=Constants.device)

    def clear(self):
        """Очищает память после обновления политики"""
        del self.states[:]
        del self.actions[:]
        del self.old_log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]
