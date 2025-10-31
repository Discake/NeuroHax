import Constants
import torch

class Memory:
    def __init__(self):
        self.states = []        # Состояния
        self.actions = []       # Выполненные действия
        self.old_log_probs = [] # Log_prob от старой политики
        self.rewards = []       # Полученные награды
        self.is_terminals = []  # Флаги окончания эпизодов
    
    def store(self, state, action, log_prob, reward, is_terminal):
        """Сохраняет один шаг траектории"""
        self.states.append(list(state))
        self.actions.append(list(action))
        self.old_log_probs.append(list(log_prob))
        self.rewards.append(list(reward))
        self.is_terminals.append(list(is_terminal))
    

    def copy_to_tensors(self):
        self.states = torch.tensor(self.states, device=Constants.device)
        self.actions = torch.tensor(self.actions, device=Constants.device)
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
