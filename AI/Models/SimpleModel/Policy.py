import torch
import torch.nn as nn

from AI.GameTranslation.TranslatorToPolicy import TranslatorToPolicy

class SimpleModel(nn.Module):
    def __init__(self, translator : TranslatorToPolicy):
        super().__init__()

        self.translator = translator
        
        hidden_dim = 32
        
        self.shared = nn.Sequential(
            nn.Linear(translator.get_state_dim(), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Выходные головы
        self.velocity_head = nn.Linear(hidden_dim // 2, 4)      # velocity mean
        self.kick_head = nn.Linear(hidden_dim // 2, 1)              # kick logit  
        
    
    def forward(self, state):    
        x = self.shared(state)
        
        velocity = self.velocity_head(x)
        
        kick_logit = self.kick_head(x)
        
        return velocity, kick_logit
    
    def get_action(self, state : dict):

        state = self.translator.translate(state)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        move_logits, kick_logit = self.forward(state)          # [B,4], [B,1]
        move_dist = torch.distributions.Bernoulli(logits=move_logits)
        kick_dist = torch.distributions.Bernoulli(logits=kick_logit)

        move = move_dist.sample()                               # [B,4]
        kick = kick_dist.sample()                               # [B,1]

        action = torch.cat([move, kick], dim=-1)                # [B,5]

        logp = move_dist.log_prob(move).sum(-1) + kick_dist.log_prob(kick).squeeze(-1)  # [B]
        entropy = move_dist.entropy().sum(-1) + kick_dist.entropy().squeeze(-1)         # [B]

        if action.size(0) == 1:
            return action.squeeze(0), logp.squeeze(0), entropy.squeeze(0)
        return action, logp, entropy
