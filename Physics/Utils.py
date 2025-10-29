import numpy as np
import torch

import Constants

def reflect_ball_from_point(ball_pos, ball_vel, point, radius):
        # ball_pos, ball_vel, point — np.array([x, y])

        # ball_pos = np.array([ball_pos.x, ball_pos.y])
        # ball_vel = np.array([ball_vel.x, ball_vel.y])
        # point = np.array([point[0], point[1]])

        direction = ball_pos - point
        dist = torch.norm(direction)
        if dist == 0:
            # редкий случай: совпадение, просто инвертировать
            return -ball_vel, point + radius * torch.tensor([1, 0]).to(Constants.device)
        if dist <= radius:
            # Нормализуем вектор
            normal = direction / dist
            # Отражаем скорость
            new_vel = ball_vel - 2 * torch.dot(ball_vel, normal) * normal
            # Сдвигаем шар на поверхность касания
            new_pos = point + normal * radius
            return new_vel, new_pos
        else:
            # Столкновения нет
            return ball_vel, ball_pos