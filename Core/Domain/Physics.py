from abc import ABC, abstractmethod
from Core.Domain.Entities.Ball import Ball
from Core.Domain.Entities.Player import Player
from Core.Domain.Entities.Wall import Wall

class Physics(ABC):
    def restrict_velocity(self, ball : Ball, max_speed):
        ball_speed = (ball.vx ** 2 + ball.vy ** 2) ** 0.5
        if ball_speed > max_speed:
            ball.vx = (ball.vx / ball_speed) * max_speed
            ball.vy = (ball.vy / ball_speed) * max_speed

    @abstractmethod
    def apply_friction(self, ball : Ball, dt : float):
        pass

    @abstractmethod
    def resolve_collision_with_ball(self, ball : Ball, ball2 : Ball):
        pass

    @abstractmethod
    def resolve_collision_with_wall(self, wall : Wall, ball : Ball):
        pass

    @abstractmethod
    def kick_ball(self, player : Player, ball : Ball, kick_power):
        pass