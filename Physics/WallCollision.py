import pygame
import torch
from Data_structure.Validable_vector import Validable_vector
from Physics.Utils import reflect_ball_from_point

class WallCollision():
    
    def __init__(self, start, end, constant, is_vertical):
        self.start = start
        self.end = end
        self.constant = constant
        self.is_vertical = is_vertical

    def detect_collision(self, object):
        """Проверка столкновения шара со стеной (нулевой толщины, конечной длины)"""
        # Для вертикальной стены x = constant, y in [start, end]
        if self.is_vertical:
            nearest_y = min(max(object.position.y, self.start), self.end)
            dx = abs(object.position.x - self.constant)
            dy = object.position.y - nearest_y
            distance2 = dx**2 + dy**2
        else:
            nearest_x = min(max(object.position.x, self.start), self.end)
            dx = object.position.x - nearest_x
            dy = abs(object.position.y - self.constant)
            distance2 = dx**2 + dy**2
        return distance2 <= object.radius**2

    def resolve_collision(self, object):
        
            # if self.detect_collision(object):
                # Координаты концов стены
                if self.is_vertical:
                    pt1 = [self.constant, self.start]
                    pt2 = [self.constant, self.end]
                    wall_normal = [1, 0]
                    wall_axis = 0
                else:
                    pt1 = [self.start, self.constant]
                    pt2 = [self.end, self.constant]
                    wall_normal = [0, 1]
                    wall_axis = 1
                
                # Центр шара
                pos = [object.position.x, object.position.y]

                # Проецируем центр шара на отрезок стены
                # Вектор от pt1 до точки шара
                wall_vec = [pt2[0]-pt1[0], pt2[1]-pt1[1]]
                ball_vec = [pos[0]-pt1[0], pos[1]-pt1[1]]
                wall_len2 = wall_vec[0]**2 + wall_vec[1]**2
                if wall_len2 == 0:
                    t = 0
                else:
                    t = (ball_vec[0]*wall_vec[0] + ball_vec[1]*wall_vec[1]) / wall_len2
                t = max(0, min(1, t))
                nearest = [pt1[0] + wall_vec[0]*t, pt1[1] + wall_vec[1]*t]

                # Проверяем, что столкновение с концом или корпусом стены
                dist2 = (pos[0]-nearest[0])**2 + (pos[1]-nearest[1])**2
                if dist2 > 0:
                    # Столкновение с концом (точкой)
                    new_vel, new_pos = reflect_ball_from_point(torch.tensor(pos), torch.stack([object.velocity.x, object.velocity.y]), torch.tensor(nearest), object.radius)
                else:
                    # Столкновение с корпусом стены (инвертируем соответствующую компоненту)
                    vel = torch.stack([object.velocity.x, object.velocity.y])
                    vel[wall_axis] *= -1
                    # Предотвращаем "залипание": отодвигаем шар на радиус от стены
                    if self.is_vertical:
                        new_pos = [self.constant + torch.sign(pos[0]-self.constant) * object.radius, pos[1]]
                    else:
                        new_pos = [pos[0], self.constant + torch.sign(pos[1]-self.constant) * object.radius]
                    new_vel = vel

                # Сохраняем результат
                object.position = Validable_vector(new_pos[0], new_pos[1])
                object.velocity = Validable_vector(new_vel[0], new_vel[1])
    
    def update(self, screen):
        self.screen = screen

    def draw(self):
        if self.is_vertical:
            x1 = x2 = self.constant
            y1 = self.start
            y2 = self.end
        else:
            y1 = y2 = self.constant
            x1 = self.start
            x2 = self.end
        # Using draw.rect module of
        # pygame to draw the line
        pygame.draw.line(self.screen, (0, 0, 0), 
                        [x1, y1], 
                        [x2, y2], 5)