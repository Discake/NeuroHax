import torch
from Objects.Ball import Ball
import Constants

class Player(Ball):
    def __init__(self, position, radius, mass, max_velocity):
        super().__init__(position, radius, mass, max_velocity)
        self.is_kicking = False
        self.velocity = torch.tensor([0., 0.], device=Constants.device)
    
    def set_kicking(self, is_kicking):
        self.is_kicking = is_kicking
        if is_kicking:
            self.set_color(Constants.kicking_color)
        else:
            self.set_color(Constants.player_color)
    
    def kick(self, ball : Ball, direction):
        if self.is_kicking:
            ball.velocity = direction * (-Constants.kicking_power)
            self.set_color(Constants.player_color)
            self.set_kicking(False)

    def __str__(self):
        return "Player " + super().__str__()
