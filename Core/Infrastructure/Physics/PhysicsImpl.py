from Core.Domain.Entities.Ball import Ball
from Core.Domain.Entities.Wall import Wall
from Core.Domain.GameConfig import PhysicsConfig
from Core.Domain.Physics import Physics
import math

from Core.Infrastructure.Physics.WallPhysics import resolve_collision as wallResolveCollision

class PhysicsImpl(Physics):
    def __init__(self, config : PhysicsConfig):
        self.config = config

    def kick_ball(self, player, ball, kick_power):
        if not player.is_kicking:
            return
        if not self.detect_collision(ball.x, ball.y, player.x, player.y, ball.radius, player.radius + self.config.kick_radius):
            return
        
        # Вычисление направления удара
        dx = ball.x - player.x
        dy = ball.y - player.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance == 0:
            return
        
        direction_x = dx / distance
        direction_y = dy / distance

        # Применение силы удара
        ball.vx += direction_x * kick_power
        ball.vy += direction_y * kick_power

        

    def apply_friction(self, ball, dt):
        acceleration = -ball.vx * self.config.friction, -ball.vy * self.config.friction
        ball.vx += acceleration[0] * dt * ball.mass
        ball.vy += acceleration[1] * dt * ball.mass

    def resolve_collision_with_ball(self, ball, ball2):
        if not self.detect_collision(ball.x, ball.y, ball2.x, ball2.y, ball.radius, ball2.radius):
            return

        ball1 = ball

        """Разрешение столкновения методом вращения векторов с учетом масс"""

        # Вектор между центрами шаров (направление удара)
        delta_x = ball2.x - ball1.x
        delta_y = ball2.y - ball1.y

        angle = math.atan2(delta_y, delta_x)

        # Матрица перехода к "ударной" системе координат
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        rotation = [
            [cos_angle, sin_angle],
            [-sin_angle, cos_angle]
        ]

        # Приведение скоростей (velocity [2]) к локальным координатам
        v1_rot = rotation[0][0] * ball1.vx + rotation[0][1] * ball1.vy, rotation[1][0] * ball1.vx + rotation[1][1] * ball1.vy
        v2_rot = rotation[0][0] * ball2.vx + rotation[0][1] * ball2.vy, rotation[1][0] * ball2.vx + rotation[1][1] * ball2.vy

        # Извлекаем массы шаров
        m1 = ball1.mass
        m2 = ball2.mass

        # Компоненты вдоль направления столкновения (axis x)
        delta_m = m1 - m2
        m_sum = m1 + m2
        M = delta_m / m_sum
        mass_mat = [
            [M, 2 * m2 / m_sum],
            [2 * m1 / m_sum, -M]
        ]

        # Классическая формула обмена импульсом с учетом масс
        v_new = mass_mat[0][0] * v1_rot[0] + mass_mat[0][1] * v2_rot[0], mass_mat[1][0] * v1_rot[0] + mass_mat[1][1] * v2_rot[0]
        v1_x_new = v_new[0]
        v2_x_new = v_new[1]
        # Перпендикулярные компоненты (axis y) не меняются
        v1_y_new = v1_rot[1]
        v2_y_new = v2_rot[1]

        # Склеиваем новые скорости в локальных координатах
        v1_rot_new = [v1_x_new, v1_y_new]
        v2_rot_new = [v2_x_new, v2_y_new]

        # Обратная матрица поворота в глобальные координаты
        rotation_inv = [
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ]

        # было:
        # v_final1 = v1_rot_new[0] * rotation_inv[0][0] + v1_rot_new[1] * rotation_inv[1][0], ...

        # стало (обратно в мир тем же rotation):
        v_final1 = (
            v1_rot_new[0] * rotation[0][0] + v1_rot_new[1] * rotation[1][0],
            v1_rot_new[0] * rotation[0][1] + v1_rot_new[1] * rotation[1][1],
        )
        v_final2 = (
            v2_rot_new[0] * rotation[0][0] + v2_rot_new[1] * rotation[1][0],
            v2_rot_new[0] * rotation[0][1] + v2_rot_new[1] * rotation[1][1],
        )

        # Обновляем скорости шаров
        ball1.vx, ball1.vy = v_final1
        ball2.vx, ball2.vy = v_final2

        # разделяем шары
        dx = ball2.x - ball1.x
        dy = ball2.y - ball1.y
        dist = math.hypot(dx, dy)
        if dist == 0:
            return
        nx, ny = dx/dist, dy/dist

        # относительная скорость (ball2 относительно ball1)
        rvx = ball2.vx - ball1.vx
        rvy = ball2.vy - ball1.vy

        # если вдоль нормали скорость "наружу" — импульс не нужен
        if rvx * nx + rvy * ny > 0:
            self.separate_balls_with_mass(ball1, ball2)
            return


    def resolve_collision_with_wall(self, wall, ball):
        wallResolveCollision(wall, ball)
    

    def detect_collision(self, x1, y1, x2, y2, radius1, radius2):
        """Проверка столкновения двух шаров"""
        # Вычисляем квадрат расстояния между центрами

        if abs(x1 - x2) > radius1 + radius2:
            return False
        if abs(y1 - y2) > radius1 + radius2:
            return False

        # Квадрат расстояния между центрами шаров
        distance_squared = (x1 - x2)** 2 + (y1 - y2) ** 2
        
        # Квадрат суммы радиусов
        radius_sum_squared = (radius1 + radius2) ** 2
        
        # Столкновение произошло, если расстояние меньше суммы радиусов
        return distance_squared < radius_sum_squared
    
    def separate_balls_with_mass(self, ball1: Ball, ball2: Ball):
        """
        Разделение перекрывающихся шаров с учетом масс:
        более тяжелый шар смещается меньше (через inverse mass).
        """
        dx = ball2.x - ball1.x
        dy = ball2.y - ball1.y

        dist2 = dx * dx + dy * dy
        if dist2 == 0.0:
            # Любое фиксированное направление, чтобы разлепить совпавшие центры
            dx, dy = 1.0, 0.0
            dist2 = 1.0

        distance = dist2 ** 0.5
        overlap = (ball1.radius + ball2.radius) - distance
        if overlap <= 0:
            return

        # Нормаль разделения
        nx = dx / distance
        ny = dy / distance

        # Обратные массы (если хотите "неподвижный" объект — mass = inf или inv_mass = 0)
        inv_m1 = 0.0 if ball1.mass == 0 else 1.0 / ball1.mass
        inv_m2 = 0.0 if ball2.mass == 0 else 1.0 / ball2.mass
        inv_sum = inv_m1 + inv_m2
        if inv_sum == 0.0:
            return

        # Величина смещения каждого шара вдоль нормали
        move1 = overlap * (inv_m1 / inv_sum)
        move2 = overlap * (inv_m2 / inv_sum)

        ball1.x -= 2 * nx * move1
        ball1.y -= 2 * ny * move1
        ball2.x += 2 * nx * move2
        ball2.y += 2 * ny * move2

