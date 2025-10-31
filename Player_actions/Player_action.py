import torch
import Constants
from Objects.Player import Player

class Player_action:
    def set_player(self, player : Player):
        self.player = player

    def act(self, input):
        vel_x = input[0]
        vel_y = input[1]

        vel_vec = torch.tensor([vel_x, vel_y]).to(Constants.device)
        self.player.velocity += vel_vec

