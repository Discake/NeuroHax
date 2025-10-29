import torch
import torch.nn as nn

import Constants

class SingleLayerNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(Constants.player_number * 2 + Constants.ball_number * 2, 8),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        probs = self.net(x)
        return probs