"""
Player - Класс игрока
"""
from Core.Domain.Entities.Ball import Ball
import torch


class Player(Ball):
    """
    Класс игрока наследуется от Ball
    Добавляет возможность удара и идентификацию
    """
    
    is_kicking: bool = False
    id: int = None
    
    def __init__(self, id, kick, x, y, mass, radius, vx, vy, kick_radius):
        super().__init__(x, y, radius, mass, vx, vy)
        self.kick_radius = kick_radius
        self.id = id
        self.is_kicking = kick
        self.is_team1 = None  # Флаг команды (устанавливается извне)
    
    def start_kick(self):
        """Начать удар"""
        self.is_kicking = True
    
    def stop_kick(self):
        """Закончить удар"""
        self.is_kicking = False
    
    def __repr__(self):
        team = "team1" if self.is_team1 else "team2"
        return f"Player(id={self.id}, team={team}, x={self.x:.2f}, y={self.y:.2f}, kicking={self.is_kicking})"