import torch
from Physics.BallCollision import BallCollision
import Constants

class Movable(BallCollision):
    def __init__(self, position, velocity, max_velocity):
        super().__init__(self)
        self.position = position
        self.velocity = velocity
        self.max_velocity = max_velocity
        self.is_collided = False
        self.drawing = None

    def validate_velocity(self):
        """Validate that the velocity does not exceed the maximum allowed speed."""
        self.velocity.validate(self.max_velocity)

    def move(self, time_increment = Constants.time_increment):
        """Move the object based on its velocity and acceleration."""
        self.resolve_position(time_increment)
        self.apply_air_resistance(time_increment)
        self.validate_velocity()

        # if(self.drawing != None):
        #     self.drawing.draw()
            
    def resolve_position(self, time):
        self.position.x = self.position.x + self.velocity.x * time
        self.position.y = self.position.y + self.velocity.y * time

    def apply_air_resistance(self, dt = Constants.time_increment):
        
        # Вычисляем текущую скорость
        if isinstance(self.velocity.x, torch.Tensor):
            # self.velocity.x = self.velocity.x.detach().clone()
            pass
        else:
            self.velocity.x = torch.tensor(self.velocity.x, dtype=torch.float32)


        if isinstance(self.velocity.y, torch.Tensor):
            # self.velocity.y = self.velocity.y.detach().clone()
            pass
        else:
            self.velocity.y = torch.tensor(self.velocity.y, dtype=torch.float32)


        speed = torch.sqrt(self.velocity.x**2 + self.velocity.y**2)
        
        if speed > 0:
            # Формула силы сопротивления: F = 0.5 * ρ * v² * Cd * A
            drag_force = Constants.friction * (0.0001 * speed * speed + 0.0001 * speed + 0.01) * self.mass # Коэффициент сопротивления воздуха
            
            # Направление противоположно скорости
            drag_x = -(self.velocity.x / speed) * drag_force
            drag_y = -(self.velocity.y / speed) * drag_force
            
            # Применяем силу
            acceleration_x = drag_x / self.mass
            acceleration_y = drag_y / self.mass
            
            self.velocity.x = self.velocity.x + acceleration_x * dt
            self.velocity.y = self.velocity.y + acceleration_y * dt
            
    def set_drawing(self, drawing):
        self.drawing = drawing