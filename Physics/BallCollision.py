import torch

class BallCollision():
    
    def __init__(self, object1):
        self.object1 = object1

    def detect_collision(self, object2):
        """Проверка столкновения двух шаров"""
        # Вычисляем квадрат расстояния между центрами
        ball1 = self.object1
        ball2 = object2

        dx = ball1.position.x - ball2.position.x
        dy = ball1.position.y - ball2.position.y
        # Квадрат расстояния между центрами шаров
        distance_squared = dx * dx + dy * dy
        
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

        """Разрешение столкновения методом вращения векторов"""
        # Угол между шарами
        dx = ball2.position.x - ball1.position.x
        dy = ball2.position.y - ball1.position.y
        angle = torch.atan2(dy, dx)
        
        # Поворачиваем векторы скоростей на угол -theta
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Повернутые скорости первого шара
        v1x_rot = ball1.velocity.x * cos_angle + ball1.velocity.y * sin_angle
        v1y_rot = ball1.velocity.y * cos_angle - ball1.velocity.x * sin_angle
        
        # Повернутые скорости второго шара
        v2x_rot = ball2.velocity.x * cos_angle + ball2.velocity.y * sin_angle
        v2y_rot = ball2.velocity.y * cos_angle - ball2.velocity.x * sin_angle
        
        # Применяем 1D формулы упругого столкновения к X-компонентам
        m1 = ball1.mass
        m2 = ball2.mass
        
        v1x_final = ((m1 - m2) * v1x_rot + 2 * m2 * v2x_rot) / (m1 + m2)
        v2x_final = ((m2 - m1) * v2x_rot + 2 * m1 * v1x_rot) / (m1 + m2)
        
        # Y-компоненты не изменяются
        v1y_final = v1y_rot
        v2y_final = v2y_rot
        
        # Поворачиваем обратно на угол +theta
        ball1.velocity.x = v1x_final * cos_angle - v1y_final * sin_angle
        ball1.velocity.y = v1y_final * cos_angle + v1x_final * sin_angle
        ball2.velocity.x = v2x_final * cos_angle - v2y_final * sin_angle
        ball2.velocity.y = v2y_final * cos_angle + v2x_final * sin_angle

        ball1.velocity *= 0.95 # затухание
        ball2.velocity *= 0.95

    def separate_balls_with_mass(self, ball2):
        """
        Разделение перекрывающихся шаров с учетом масс
        Более тяжелый шар смещается меньше
        """
        # Вектор от центра ball1 к ball2

        ball1 = self.object1

        dx = ball2.position.x - ball1.position.x
        dy = ball2.position.y - ball1.position.y
        distance = torch.sqrt(dx * dx + dy * dy)
        
        # Предотвращаем деление на ноль
        if distance == 0:
            distance = 0.0001
            dx = 0.0001
        
        # Величина перекрытия
        overlap = (ball1.radius + ball2.radius) - distance
        
        if overlap > 0:
            # Нормализованный вектор разделения
            nx = dx * 2 / distance
            ny = dy * 2 / distance
            
            # Распределяем смещение пропорционально массам
            # Более тяжелый объект движется меньше
            total_mass = ball1.mass + ball2.mass
            ratio1 = ball2.mass / total_mass
            ratio2 = ball1.mass / total_mass
            
            # Смещаем каждый шар
            ball1.position.x = ball1.position.x - overlap * ratio1 * nx 
            ball1.position.y = ball1.position.y - overlap * ratio1 * ny
            ball2.position.x = ball2.position.x + overlap * ratio2 * nx
            ball2.position.y = ball2.position.y + overlap * ratio2 * ny 