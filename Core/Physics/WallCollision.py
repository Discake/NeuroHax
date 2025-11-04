import torch
import Constants
from Core.Physics.Utils import reflect_ball_from_point
from Core.Objects.Ball import Ball

class WallCollision():
    
    def __init__(self, start, end, constant, is_vertical):
        # Сразу конвертируем в Python float
        self.start = float(start)
        self.end = float(end) 
        self.constant = float(constant)
        self.is_vertical = is_vertical

        

    def detect_collision(self, object : Ball):
        """Проверка столкновения шара со стеной (нулевой толщины, конечной длины)"""

        with torch.no_grad():
            # Для вертикальной стены x = constant, y in [start, end]
            if self.is_vertical:
                if abs(object.position[0] - self.constant) > object.radius:
                    return False

                nearest_y = min(max(object.position[1], self.start), self.end)
                xy = torch.tensor([self.constant, nearest_y], device=Constants.device)
                xy = object.position - xy
            else:
                if abs(object.position[1] - self.constant) > object.radius:
                    return False

                nearest_x = min(max(object.position[0], self.start), self.end)
                xy = torch.tensor([nearest_x, self.constant], device=Constants.device)
                xy = object.position - xy
            distance2 = torch.dot(xy, xy)
            return distance2 <= object.radius**2
        
    def detect_collision_pure_python(self, obj):
        """Чистый Python без PyTorch операций"""
        # Извлекаем координаты один раз в начале функции
        obj_pos = obj.position
        pos_x = float(obj_pos[0])  # Быстрая конвертация
        pos_y = float(obj_pos[1])
        radius_sq = obj.radius * obj.radius  # Python float операция
        
        if self.is_vertical:
            # Зажимаем Y без функций min/max - используем if
            if pos_y < self.start:
                nearest_y = self.start
            elif pos_y > self.end:
                nearest_y = self.end
            else:
                nearest_y = pos_y
                
            dx = pos_x - self.constant
            dy = pos_y - nearest_y
        else:
            # Зажимаем X аналогично
            if pos_x < self.start:
                nearest_x = self.start
            elif pos_x > self.end:
                nearest_x = self.end
            else:
                nearest_x = pos_x
                
            dx = pos_x - nearest_x
            dy = pos_y - self.constant
            
        return dx*dx + dy*dy <= radius_sq


    def resolve_collision(self, object : Ball):
        
            # if self.detect_collision(object):
                # Координаты концов стены
                if self.is_vertical:
                    pt1 = torch.tensor([self.constant, self.start], device=Constants.device)
                    pt2 = torch.tensor([self.constant, self.end], device=Constants.device)
                    wall_axis = 0
                else:
                    pt1 = torch.tensor([self.start, self.constant], device=Constants.device)
                    pt2 = torch.tensor([self.end, self.constant], device=Constants.device)
                    wall_axis = 1
                
                # Центр шара
                pos = object.position

                # Проецируем центр шара на отрезок стены
                # Вектор от pt1 до точки шара
                # wall_vec = [pt2[0]-pt1[0], pt2[1]-pt1[1]]
                wall_vec = pt2 - pt1
                # ball_vec = [pos[0]-pt1[0], pos[1]-pt1[1]]
                ball_vec = pos - pt1
                # wall_len2 = wall_vec[0]**2 + wall_vec[1]**2
                wall_len2 = torch.dot(wall_vec, wall_vec)
                if wall_len2 == 0:
                    t = 0
                else:
                    # t = (ball_vec[0]*wall_vec[0] + ball_vec[1]*wall_vec[1]) / wall_len2
                    t = torch.dot(ball_vec, wall_vec) / wall_len2
                t = max(0, min(1, t))
                # nearest = [pt1[0] + wall_vec[0]*t, pt1[1] + wall_vec[1]*t]
                nearest = pt1 + wall_vec * t

                # Проверяем, что столкновение с концом или корпусом стены
                # dist2 = (pos[0]-nearest[0])**2 + (pos[1]-nearest[1])**2
                dist2 = torch.dot(pos - nearest, pos - nearest)
                if dist2 > 0:
                    # Столкновение с концом (точкой)
                    new_vel, new_pos = reflect_ball_from_point(pos, object.velocity, nearest, object.radius)
                else:
                    # Столкновение с корпусом стены (инвертируем соответствующую компоненту)
                    vel = object.velocity
                    vel[wall_axis] *= -1
                    # Предотвращаем "залипание": отодвигаем шар на радиус от стены
                    if self.is_vertical:
                        new_pos = torch.tensor([self.constant + torch.sign(pos[0]-self.constant) * object.radius, pos[1]], device=Constants.device)
                    else:
                        new_pos = torch.tensor([pos[0], self.constant + torch.sign(pos[1]-self.constant) * object.radius], device=Constants.device)
                    new_vel = vel

                # Сохраняем результат
                object.position = new_pos
                object.velocity = new_vel
    
    def update(self, screen):
        self.screen = screen