from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, states, actions, rewards, log_probs):
        self.states = [s.detach() for s in states]
        self.actions = [a.detach() for a in actions]
        self.rewards = rewards
        self.log_probs = log_probs

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'log_prob': self.log_probs[idx]
        }
