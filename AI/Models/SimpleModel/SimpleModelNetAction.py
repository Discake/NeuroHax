"""
Net_action для SimpleModel - преобразование бинарных действий в движения
"""

import torch
import math
import Constants
from Core.Domain.Entities.Ball import Ball
from AI.Models.SimpleModel.SimpleModelTranslator import SimpleModelTranslator


class SimpleModelNetAction:
    """
    Класс для преобразования выходов SimpleModel в действия игрока
    SimpleModel возвращает 5 бинарных значений: [up, down, left, right, kick]
    """

    def __init__(self, map, net, player, is_team_1=True):
        """
        Инициализация SimpleModelNetAction

        Args:
            map: Объект карты с игроками и мячами
            net: SimpleModel политика
            player: Игрок, которым управляем
            is_team_1: Принадлежность к команде 1 (True) или 2 (False)
        """
        self.map = map
        self.net = net
        self.player = player
        self.is_team_1 = is_team_1

        # Инициализация транслятора входных данных
        self.translator = SimpleModelTranslator(map, player, is_team_1)

        # Параметры действия
        self.action = torch.zeros(5, device=Constants.device)  # 4 направления + kick

        # Параметры удара
        self.kick_power = 216.0
        self.kick_cooldown = 0
        self.kick_cooldown_max = 6   # 4× более частые удары (было 25)

        # Параметры движения
        self.move_power = 1.6  # Сила движения для бинарных команд (УВЕЛИЧЕНО В 2 РАЗА)

    def act(self, action):
        """
        Применение действия от SimpleModel

        Args:
            action: Тензор действия [up, down, left, right, kick] (binary)
        """
        self.action = action

        # Сбрасываем флаг удара в начале каждого шага
        self.player.is_kicking = False

        # Извлечение компонентов действия
        if action.dim() == 1:
            up = action[0].item() if len(action) > 0 else 0
            down = action[1].item() if len(action) > 1 else 0
            left = action[2].item() if len(action) > 2 else 0
            right = action[3].item() if len(action) > 3 else 0
            kick = action[4].item() if len(action) > 4 else 0
        else:
            up = action[0, 0].item()
            down = action[0, 1].item()
            left = action[0, 2].item()
            right = action[0, 3].item()
            kick = action[0, 4].item()

        # Применение движения (бинарные направления)
        self._apply_binary_movement(up, down, left, right)

        # Применение удара
        if kick > 0.5:
            self._apply_kick()

        # Обновление кулдауна
        if self.kick_cooldown > 0:
            self.kick_cooldown -= 1

    def _apply_binary_movement(self, up, down, left, right):
        """
        Применение бинарного движения к игроку

        Args:
            up, down, left, right: Бинарные команды (0 или 1)
        """
        # Вычисляем направление движения
        dx = 0.0
        dy = 0.0

        if up:
            dy -= 1.0
        if down:
            dy += 1.0
        if left:
            dx -= 1.0
        if right:
            dx += 1.0

        # Нормализуем диагональное движение
        length = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            dx /= length
            dy /= length

        # Преобразуем в абсолютную скорость
        # Для team2 x-ось перевёрнута в состоянии (flip=-1), поэтому и действие нужно перевернуть
        x_flip = 1.0 if self.is_team_1 else -1.0
        target_vx = dx * x_flip * Constants.max_player_speed * self.move_power
        target_vy = dy * Constants.max_player_speed * self.move_power

        # Плавное ускорение
        self.player.vx += (target_vx - self.player.vx) * 0.3
        self.player.vy += (target_vy - self.player.vy) * 0.3

        # Применение трения
        self.player.vx *= 0.95
        self.player.vy *= 0.95

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

        kick_range = 30.0  # > 22 (sum of radii) — иначе физика выталкивает мяч за зону удара

        if distance > kick_range:
            return

        # Направление удара — строго по линии центров игрок→мяч
        if distance > 0:
            kick_dir_x = dx / distance
            kick_dir_y = dy / distance
        else:
            kick_dir_x = 1.0 if self.is_team_1 else -1.0
            kick_dir_y = 0.0

        # Импульс: сила удара + проекция скорости игрока на направление удара
        player_contrib = self.player.vx * kick_dir_x + self.player.vy * kick_dir_y
        impulse = self.kick_power + player_contrib
        ball.vx += kick_dir_x * impulse
        ball.vy += kick_dir_y * impulse

        # Установка флага удара
        self.player.is_kicking = True

        # Установка кулдауна
        self.kick_cooldown = self.kick_cooldown_max

    def _get_nearest_ball(self):
        """
        Получение ближайшего мяча
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
        """
        return self.action

    def reset(self):
        """Сброс состояния после эпизода"""
        self.action = torch.zeros(5, device=Constants.device)
        self.kick_cooldown = 0
        self.player.is_kicking = False
