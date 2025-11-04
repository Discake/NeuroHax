import torch

import Constants

class BallCollision():
    
    def __init__(self, object1):
        self.object1 = object1

    def detect_collision(self, object2):
        """Проверка столкновения двух шаров"""
        # Вычисляем квадрат расстояния между центрами
        ball1 = self.object1
        ball2 = object2

        if torch.abs(ball1.position[0] - ball2.position[0]) > ball1.radius + ball2.radius:
            return False
        if torch.abs(ball1.position[1] - ball2.position[1]) > ball1.radius + ball2.radius:
            return False

        # Квадрат расстояния между центрами шаров
        distance_squared = torch.dot(ball1.position - ball2.position, ball1.position - ball2.position)
        
        # Квадрат суммы радиусов
        radius_sum_squared = (ball1.radius + ball2.radius) ** 2
        
        # Столкновение произошло, если расстояние меньше суммы радиусов
        return distance_squared < radius_sum_squared

    def resolve_collision(self, object2):
        # if(self.detect_collision(object2)):
        self.resolve_collision_rotation(object2)
        self.separate_balls_with_mass(object2)
        
    def resolve_collision_rotation(self, ball2):
        ball1 = self.object1

        """Разрешение столкновения методом вращения векторов с учетом масс"""

        # Вектор между центрами шаров (направление удара)
        delta = ball2.position - ball1.position  # torch.tensor([dx, dy])
        angle = torch.atan2(delta[1], delta[0])

        # Матрица перехода к "ударной" системе координат
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        rotation = torch.tensor([
            [cos_angle, sin_angle],
            [-sin_angle, cos_angle]
        ], device=Constants.device)

        # Приведение скоростей (velocity [2]) к локальным координатам
        v1_rot = torch.matmul(rotation, ball1.velocity)
        v2_rot = torch.matmul(rotation, ball2.velocity)

        # Извлекаем массы шаров
        m1 = ball1.mass
        m2 = ball2.mass

        # Компоненты вдоль направления столкновения (axis x)
        delta_m = m1 - m2
        m_sum = m1 + m2
        M = delta_m / m_sum
        mass_mat = torch.tensor([
            [M, 2 * m2 / m_sum],
            [2 * m1 / m_sum, -M]
        ], device=Constants.device)

        # Классическая формула обмена импульсом с учетом масс
        v_new = torch.matmul(mass_mat, torch.tensor([v1_rot[0], v2_rot[0]], device=Constants.device))
        v1_x_new = v_new[0]
        v2_x_new = v_new[1]
        # Перпендикулярные компоненты (axis y) не меняются
        v1_y_new = v1_rot[1]
        v2_y_new = v2_rot[1]

        # Склеиваем новые скорости в локальных координатах
        v1_rot_new = torch.tensor([v1_x_new, v1_y_new], device=Constants.device)
        v2_rot_new = torch.tensor([v2_x_new, v2_y_new], device=Constants.device)

        # Обратная матрица поворота в глобальные координаты
        rotation_inv = torch.tensor([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ], device=Constants.device)

        # Преобразуем обратно
        ball1.velocity = torch.matmul(rotation_inv, v1_rot_new)
        ball2.velocity = torch.matmul(rotation_inv, v2_rot_new)

    def separate_balls_with_mass(self, ball2):
        """
        Разделение перекрывающихся шаров с учетом масс
        Более тяжелый шар смещается меньше
        """
        # Вектор от центра ball1 к ball2

        ball1 = self.object1

        d = ball2.position - ball1.position
        distance = d.norm()
        
        # Предотвращаем деление на ноль
        if distance == 0:
            distance = 0.0001
        
        # Величина перекрытия
        overlap = (ball1.radius + ball2.radius) - distance
        
        if overlap > 0:
            # Нормализованный вектор разделения
            nx = d[0] / distance
            ny = d[1] / distance
            
            # Распределяем смещение пропорционально массам
            # Более тяжелый объект движется меньше
            total_mass = ball1.mass + ball2.mass
            ratio1 = ball2.mass / total_mass
            ratio2 = ball1.mass / total_mass

            overlap_vector = torch.tensor([overlap * nx, overlap * ny], device=Constants.device)
            
            # Смещаем каждый шар
            ball1.position = ball1.position - ratio1 * overlap_vector * 2
            ball2.position = ball2.position + ratio2 * overlap_vector * 2