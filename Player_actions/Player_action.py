import torch
import Constants
from Core.Objects.Player import Player

class Player_action:
    def set_player(self, player : Player):
        self.player = player

    def act(self, input):
        with torch.no_grad():

            temp = input.squeeze(0)

            self.player.velocity[0] = Constants.acceleration * temp[0]
            self.player.velocity[1] = Constants.acceleration * temp[1]
            self.player.set_kicking(temp[2] > 0)

