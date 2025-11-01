import torch
import Constants
from Core.Objects.Player import Player

class Player_action:
    def set_player(self, player : Player):
        self.player = player

    def act(self, input):
        vel_x = input[0]
        vel_y = input[1]
        kick = input[2]

        vel_vec = torch.tensor([vel_x, vel_y]).to(Constants.device)
        self.player.acceleration = Constants.acceleration * vel_vec
        self.player.set_kicking(kick > 0)

