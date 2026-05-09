"""
Константы для проекта NeuroHax
"""
import torch

# Размеры поля (физический мир: мяч, ворота, расстановка)
field_size = (600, 420)

# Отступ поля от края окна (px)
field_offset_x = 120
field_offset_y = 120

# Размер окна = поле + отступы
window_size = (field_size[0] + 2 * field_offset_x,
               field_size[1] + 2 * field_offset_y)   # (720, 480)

# Центры ПОЛЯ в мировых координатах (не окна)
x_center = field_size[0] // 2   # 300
y_center = field_size[1] // 2   # 210
right_gates_center_x = field_size[0]  # 600
left_gates_center_x  = 0

# Параметры игры
player_number = 1
ball_number = 1
max_player_speed = 150.0  # Увеличено в 9 раз от оригинала (было 5.0)
max_ball_speed = 1500.0   # Соответствует kick_power (1350) + запас

# Размеры для нейросети
state_size = 19
action_size = 5  # velocity_x, velocity_y, kick

# Устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Цвета (для отрисовки)
colors = {
    'background_color': (30, 30, 30),
    'field_color': (50, 50, 50),
    'ball_color': (255, 255, 255),
    'player_red': (255, 50, 50),
    'player_blue': (50, 50, 255)
}
