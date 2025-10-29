class Memory:
    def __init__(self):
        self.states = []        # Состояния
        self.actions = []       # Выполненные действия
        self.old_log_probs = [] # Log_prob от старой политики
        self.rewards = []       # Полученные награды
        self.is_terminals = []  # Флаги завершения эпизода
    
    def store(self, state, action, log_prob, reward, is_terminal):
        """Сохраняет один шаг траектории"""
        self.states.append(state.squeeze(0))
        self.actions.append(action.squeeze(0))
        self.old_log_probs.append(log_prob.squeeze(0))
        self.rewards.append(reward.squeeze(0))
        self.is_terminals.append(is_terminal)
    
    def clear(self):
        """Очищает память после обновления политики"""
        del self.states[:]
        del self.actions[:]
        del self.old_log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]
