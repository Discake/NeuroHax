import torch
import Constants
from Core.Objects.Player import Player

class Player_action:
    def set_player(self, player : Player):
        self.player = player
        
    def act(self, input):
        with torch.no_grad():
            vel_x = input[0]
            vel_y = input[1]
            kick = input[2]

            self.player.acceleration[0] = Constants.acceleration * vel_x
            self.player.acceleration[1] = Constants.acceleration * vel_y
            self.player.set_kicking(kick > 0)

