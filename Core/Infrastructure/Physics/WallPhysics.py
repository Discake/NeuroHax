from Core.Domain.Entities.Ball import Ball
from Core.Domain.Entities.Wall import Wall


def detect_wall_collision(wall : Wall, ball : Ball):
    """Чистый Python без PyTorch операций"""
    
    pos_x = ball.x  # Быстрая конвертация
    pos_y = ball.y
    radius_sq = ball.radius * ball.radius  # Python float операция
    
    if pos_y < wall.y:
        nearest_y = wall.y
    elif pos_y > wall.y + wall.height:
        nearest_y = wall.y + wall.height
    else:
        nearest_y = pos_y
    # Зажимаем X аналогично
    if pos_x < wall.x:
        nearest_x = wall.x
    elif pos_x > wall.x + wall.width:
        nearest_x = wall.x + wall.width
    else:
        nearest_x = pos_x
        
    dx = pos_x - nearest_x
    dy = pos_y - nearest_y
        
    return dx*dx + dy*dy <= radius_sq
    
def resolve_collision(wall : Wall, ball : Ball):
    if not detect_wall_collision(wall, ball):
        return

    # if self.detect_collision(object):
    # Координаты концов стены
    
    pt1 = wall.x, wall.y
    pt2 = wall.x + wall.width, wall.y + wall.height
    
    # Центр шара
    pos = ball.x, ball.y
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
    new_vel, new_pos = reflect_ball_from_point(pos, [ball.vx, ball.vy], nearest, ball.radius)
    # Сохраняем результат
    ball.x, ball.y = new_pos
    ball.vx, ball.vy = new_vel

def reflect_ball_from_point(ball_pos, ball_vel, point, radius):
    direction = ball_pos[0] - point[0], ball_pos[1] - point[1]

    dist = (direction[0]**2 + direction[1]**2) ** 0.5
    if dist <= radius:
        # Нормализуем вектор
        normal = direction[0] / dist, direction[1] / dist

        dot = ball_vel[0] * normal[0] + ball_vel[1] * normal[1]
        # Отражаем скорость
        new_vel = ball_vel[0] - 2 * dot * normal[0], ball_vel[1] - 2 * dot * normal[1]
        # Сдвигаем шар на поверхность касания
        new_pos = point[0] + normal[0] * radius, point[1] + normal[1] * radius
        return new_vel, new_pos
    else:
        # Столкновения нет
        return ball_vel, ball_pos