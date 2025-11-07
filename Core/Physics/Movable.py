import math
import torch
from Core.Physics.BallCollision import BallCollision
import Constants

class Movable(BallCollision):
    def __init__(self, position : torch.Tensor, velocity : torch.Tensor, acceleration : torch.Tensor, max_velocity):
        super().__init__(self)
        self.position = position
        self.velocity = velocity
        self.acceleration=acceleration
        self.max_velocity = max_velocity
        self.is_collided = False
        self.drawing = None

    def validate_velocity(self):
        """Validate that the velocity does not exceed the maximum allowed speed."""
        vel_norm2 = torch.dot(self.velocity, self.velocity)
        if( vel_norm2 > self.max_velocity ** 2):
            self.velocity = self.velocity * (self.max_velocity / math.sqrt(vel_norm2))

    def move(self, time_increment = Constants.time_increment):
        """Move the object based on its velocity and acceleration."""
        self.resolve_position(time_increment)
        self.validate_velocity()
        self.apply_air_resistance(time_increment)
            
    def resolve_position(self, time):
        self.position = self.position.add_(self.velocity * time)
        self.velocity = self.velocity.add_(self.acceleration * time)

    def apply_air_resistance(self, time):
        
        # Вычисляем текущую скорость
        speed = self.velocity.norm()
        
        if speed > 0:
            # Формула силы сопротивления:
            # drag_force = Constants.friction * (0.0001 * speed + 0.001 + 0.0001)
            
            # # Применяем силу
            # acceleration = -self.velocity * drag_force * self.mass

            acceleration = -self.velocity * Constants.friction
            
            self.velocity += acceleration * time
            
    def set_drawing(self, drawing):
        self.drawing = drawing