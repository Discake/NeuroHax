import torch
from Physics.Movable import Movable
import Constants
from Data_structure.Validable_vector import Validable_vector

class Ball (Movable):
    def __init__(self, position, radius, mass, max_velocity):
        init_velocity = Validable_vector(torch.tensor(0.).to(Constants.device), torch.tensor(0.).to(Constants.device))
        super().__init__(position, velocity=init_velocity, max_velocity=max_velocity)
        self.radius = radius
        self.mass = mass
        self.color = (255, 0, 0)  # Default color red
        
    def set_color(self, color):
        """Set the color of the ball."""
        self.color = color

    def __str__(self):
        return f"Ball(position.x={self.position.x}, position.y={self.position.y}, velocity.x={self.velocity.x}, velocity.y={self.velocity.y})"