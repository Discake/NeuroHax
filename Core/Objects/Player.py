import torch
from Core.Objects.Ball import Ball
import Constants

class Player(Ball):
    def __init__(self, position, radius, mass, max_velocity, is_team1):
        super().__init__(position, radius, mass, max_velocity)
        self.is_kicking = False
        self.velocity = torch.tensor([0., 0.], device=Constants.device)

        if is_team1:
            self.color = Constants.player_color1
            self.kicking_color = Constants.kicking_color1
        else:
            self.color = Constants.player_color2
            self.kicking_color = Constants.kicking_color2
    
    def set_kicking(self, is_kicking):
        self.is_kicking = is_kicking
        if is_kicking:
            self.set_color(self.kicking_color)
        else:
            self.set_color(self.color)
    
    def kick(self, ball : Ball, direction):
        if self.is_kicking:
            ball.velocity = direction * (-Constants.kicking_power)
            self.set_color(Constants.player_color)
            self.set_kicking(False)

    def __str__(self):
        return "Player " + super().__str__()
