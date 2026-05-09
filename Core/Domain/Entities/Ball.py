"""
Ball - Базовый класс мяча
"""
import torch


class Ball():
    """
    Класс мяча с позицией, скоростью и физическими параметрами
    """
    
    def __init__(self, x, y, radius, mass, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.mass = mass
    
    @property
    def position(self):
        """Позиция мяча как torch tensor"""
        return torch.tensor([self.x, self.y], dtype=torch.float32)
    
    @position.setter
    def position(self, value):
        """Установка позиции из torch tensor"""
        self.x = value[0].item()
        self.y = value[1].item()
    
    @property
    def velocity(self):
        """Скорость мяча как torch tensor"""
        return torch.tensor([self.vx, self.vy], dtype=torch.float32)
    
    @velocity.setter
    def velocity(self, value):
        """Установка скорости из torch tensor"""
        self.vx = value[0].item()
        self.vy = value[1].item()
    
    def move(self, dt):
        """
        Перемещение мяча
        
        Args:
            dt: Дельта времени
        """
        self.x += self.vx * dt
        self.y += self.vy * dt
    
    def __repr__(self):
        return f"Ball(x={self.x:.2f}, y={self.y:.2f}, vx={self.vx:.2f}, vy={self.vy:.2f})"