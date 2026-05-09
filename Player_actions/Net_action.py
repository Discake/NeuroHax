"""
Net_action - Преобразование действий нейросети в движения игрока
Реализует физику взаимодействия игрока с мячом
"""

import torch
import math
import Constants
from Core.Domain.Entities.Ball import Ball
from Core.Domain.Entities.Player import Player
from AI.Translator import Translator


class Net_action:
    """
    Класс для преобразования выходных данных нейросети в действия игрока
    и взаимодействия с мячом (удары, движение)
    """
    
    def __init__(self, map, net, player, is_team_1=True):
        """
        Инициализация Net_action
        
        Args:
            map: Объект карты с игроками и мячами
            net: Нейросеть (политика) для принятия решений
            player: Игрок, которым управляем
            is_team_1: Принадлежность к команде 1 (True) или 2 (False)
        """
        self.map = map
        self.net = net
        self.player = player
        self.is_team_1 = is_team_1
        
        # Инициализация транслятора входных данных
        self.translator = Translator(map, net, player, is_team_1)
        
        # Параметры действия
        self.action = torch.zeros(3, device=Constants.device)
        
        # Параметры удара
        self.kick_power = 8.0  # Увеличена сила удара для более заметного ускорения мяча
        self.kick_cooldown = 0  # Кулдаун удара
        self.kick_cooldown_max = 3  # Уменьшено для более частых ударов
        
        # Параметры движения
        self.max_acceleration = 0.5  # Максимальное ускорение
        self.friction = 0.95  # Трение
    
    def act(self, action):
        """
        Применение действия от нейросети
        
        Args:
            action: Тензор действия [velocity_x, velocity_y, kick]
        """
        self.action = action
        
        # Извлечение компонентов действия
        if action.dim() == 1:
            velocity_x = action[0].item() if len(action) > 0 else 0.0
            velocity_y = action[1].item() if len(action) > 1 else 0.0
            kick = action[2].item() if len(action) > 2 else 0.0
        else:
            velocity_x = action[0, 0].item()
            velocity_y = action[0, 1].item()
            kick = action[0, 2].item() if action.shape[1] > 2 else 0.0
        
        # Применение движения
        self._apply_movement(velocity_x, velocity_y)
        
        # Применение удара
        if kick > 0.5:  # Порог для удара
            self._apply_kick()
        
        # Обновление кулдауна
        if self.kick_cooldown > 0:
            self.kick_cooldown -= 1
    
    def _apply_movement(self, velocity_x, velocity_y):
        """
        Применение движения к игроку
        
        Args:
            velocity_x: Желаемая скорость по X (нормализованная -1..1)
            velocity_y: Желаемая скорость по Y (нормализованная -1..1)
        """
        # Ограничение скорости
        velocity_x = max(-1.0, min(1.0, velocity_x))
        velocity_y = max(-1.0, min(1.0, velocity_y))
        
        # Преобразование в абсолютную скорость
        target_vx = velocity_x * Constants.max_player_speed
        target_vy = velocity_y * Constants.max_player_speed
        
        # Плавное ускорение к целевой скорости
        self.player.vx += (target_vx - self.player.vx) * self.max_acceleration
        self.player.vy += (target_vy - self.player.vy) * self.max_acceleration
        
        # Применение трения
        self.player.vx *= self.friction
        self.player.vy *= self.friction
        
        # Ограничение максимальной скорости
        speed = math.sqrt(self.player.vx ** 2 + self.player.vy ** 2)
        if speed > Constants.max_player_speed:
            self.player.vx = (self.player.vx / speed) * Constants.max_player_speed
            self.player.vy = (self.player.vy / speed) * Constants.max_player_speed
    
    def _apply_kick(self):
        """
        Применение удара по мячу
        """
        # Проверка кулдауна
        if self.kick_cooldown > 0:
            return
        
        # Находим ближайший мяч
        ball = self._get_nearest_ball()
        if ball is None:
            return
        
        # Проверка расстояния до мяча
        dx = ball.x - self.player.x
        dy = ball.y - self.player.y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        # Радиус взаимодействия (УВЕЛИЧЕН для более лёгкого попадания)
        kick_range = self.player.radius + ball.radius + 10.0  # Было +5.0
        
        if distance > kick_range:
            return
        
        # Направление удара (от игрока к мячу)
        if distance > 0:
            kick_dir_x = dx / distance
            kick_dir_y = dy / distance
        else:
            # Если игрок и мяч в одной точке, бьём в направлении движения игрока
            player_speed = math.sqrt(self.player.vx ** 2 + self.player.vy ** 2)
            if player_speed > 0:
                kick_dir_x = self.player.vx / player_speed
                kick_dir_y = self.player.vy / player_speed
            else:
                # По умолчанию бьём вправо для team1 и влево для team2
                kick_dir_x = 1.0 if self.is_team_1 else -1.0
                kick_dir_y = 0.0
        
        # Применение силы удара к мячу
        ball.vx += kick_dir_x * self.kick_power
        ball.vy += kick_dir_y * self.kick_power
        
        # Установка флага удара
        self.player.is_kicking = True
        
        # Установка кулдауна
        self.kick_cooldown = self.kick_cooldown_max
    
    def _get_nearest_ball(self):
        """
        Получение ближайшего мяча
        
        Returns:
            Ball: Ближайший мяч или None
        """
        if not hasattr(self.map, 'balls') or len(self.map.balls) == 0:
            return None
        
        nearest_ball = None
        min_distance = float('inf')
        
        for ball in self.map.balls:
            dx = ball.x - self.player.x
            dy = ball.y - self.player.y
            distance = math.sqrt(dx ** 2 + dy ** 2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_ball = ball
        
        return nearest_ball
    
    def get_action(self):
        """
        Получение текущего действия
        
        Returns:
            torch.Tensor: Текущее действие [velocity_x, velocity_y, kick]
        """
        return self.action
    
    def reset(self):
        """Сброс состояния после эпизода"""
        self.action = torch.zeros(3, device=Constants.device)
        self.kick_cooldown = 0
        self.player.is_kicking = False
    
    def set_kick_power(self, power):
        """
        Установка силы удара
        
        Args:
            power: Сила удара
        """
        self.kick_power = power
    
    def set_kick_cooldown(self, cooldown):
        """
        Установка кулдауна удара
        
        Args:
            cooldown: Кулдаун в тиках
        """
        self.kick_cooldown_max = cooldown
