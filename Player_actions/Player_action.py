import torch
import Constants
from Core.Objects.Player import Player

class Player_action:
    def set_player(self, player : Player):
        self.player = player

    def act(self, input):
        with torch.no_grad():

            temp = input.squeeze(0)

            if hasattr(self, "is_team1"):
                if not self.is_team1:
                    temp[0] = -temp[0]

            self.player.velocity[0] += temp[0] * Constants.acceleration
            self.player.velocity[1] += temp[1] * Constants.acceleration
            self.player.set_kicking(temp[2] > 0)

