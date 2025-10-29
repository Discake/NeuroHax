import torch
from Data_structure.Validable_vector import Validable_vector

# Physics

player_radius = 30  # Radius of the player
ball_radius = 30  # Radius of the ball
player_mass = 300.0  # Mass of the player
ball_mass = 300  # Mass of the ball
max_player_speed = 100  # Maximum speed of the player
max_ball_speed = 6.0  # Maximum speed of the ball
time_increment = 1  # Time increment for movement
speed_increment = 3 # Speed increment for the player

iterations = 1 * round(time_increment)

friction = 10  # Friction coefficient

kicking_power = 35  # Power of the kick
kick_radius = 15  # Radius of the kick

# Color
player_color = (25, 25, 112)  # Blue color for the player
ball_color = (255, 69, 0)  # Red color for the ball
filling_color = (255, 255, 255)  # White color for the background
field_color = (60, 179, 113)  # Green color for the field
main_player_color = (255, 255, 0)  # Red color for the main player
gates_outer_color = (127, 127, 127)  # Grey color for the outer gates
kicking_color = (255, 105, 180)  # Red color for kicking
gates_inner_color = (0, 100, 0)

# Amount of players and balls
# player_number = int(input("Enter the number of players (1 or 2): "))  # Player number
player_number = 1  # Player number
ball_number = 1  # Ball number

# Window size
window_size = (1000, 800)  # Size of the game window

field_margin = 200

field_size = (window_size[0] - field_margin, window_size[1] - field_margin)  # Size of the field

# Positions of players and balls
player_x_position_from_center_team1 = -100  # X position of the player from the center of the field
player_x_position_from_center_team2 = 100  # X position of the player from the center of the field
ball_x_position_from_center = 0  # X position of the ball from the center of the field

x_center = window_size[0] // 2  # X coordinate of the center of the window
y_center = window_size[1] // 2  # Y coordinate of the center of the window

player_y_increment = (player_number - 1) * 1 * player_radius  # Y position increment for each player
player_y_position_from_center = 0  # Y position of the player from the center of the field
ball_y_increment = (ball_number - 1) * 3 * ball_radius  # Y position increment for each ball
ball_y_position_from_center = 0  # Y position of the ball from the center of the field



players_y_positions = []
balls_y_positions = []

for i in range(player_number):
    players_y_positions.append(player_y_position_from_center + player_y_increment * (i + 1) * (-1 if i % 2 == 0 else 1))
    balls_y_positions.append(ball_y_position_from_center + ball_y_increment * (i + 1) * (-1 if i % 2 == 0 else 1))

players_positions_team1 = [Validable_vector(torch.tensor(player_x_position_from_center_team1 + x_center, dtype=torch.float32), torch.tensor(y + y_center, dtype=torch.float32)) for y in players_y_positions]
players_positions_team2 = [Validable_vector(torch.tensor(player_x_position_from_center_team2 + x_center, dtype=torch.float32), torch.tensor(y + y_center, dtype=torch.float32)) for y in players_y_positions]
balls_positions = [Validable_vector(torch.tensor(ball_x_position_from_center + x_center, dtype=torch.float32), torch.tensor(y + y_center, dtype=torch.float32)) for y in balls_y_positions]

# Gates positions

left_gates_center_x = 0
left_gates_center_y = y_center


gates_outer_width = 100
gates_outer_height = 300
gates_inner_width = 80
gates_inner_height = 250

right_gates_center_x = window_size[0] - gates_outer_width
right_gates_center_y = y_center

# Проверка, что все переменные определены
# field_size = [width, height]
# gates_inner_height - высота ворот
# gates_outer_width - ширина ворот
# field_separation - отступ/шаг

def set_wall(x, y, width, height, is_vertical):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    class Wall:
        def __init__(self, is_vertical):
            self.start = None
            self.end = None
            self.constant = None
            self.is_vertical = is_vertical

    wall = Wall(is_vertical)
    wall.start = Validable_vector(x, y)
    wall.end = Validable_vector(x + width, y + height)

    wall.start = wall.start.y if wall.is_vertical else wall.start.x
    wall.end = wall.end.y if wall.is_vertical else wall.end.x
    wall.constant = x if wall.is_vertical else y
    return wall

wall1 = set_wall(field_margin / 2, field_margin / 2 + field_size[1] / 2 + gates_inner_height / 2, field_size[0], field_size[1] / 2 - gates_inner_height / 2, True) #left
wall2 = set_wall(field_margin / 2, field_margin / 2, field_size[0], field_size[1] / 2 - gates_inner_height / 2, True) #left ok


wall3 = set_wall(field_margin / 2 + field_size[0], field_margin / 2, field_size[0] / 2 + gates_inner_width / 2, field_size[1] / 2 - gates_inner_height / 2, True) #right ok
wall4 = set_wall(field_margin / 2 + field_size[0], field_margin / 2 + field_size[1] / 2 + gates_inner_height / 2, field_size[0] / 2 - gates_inner_width / 2, field_size[1] / 2 - gates_inner_height / 2, True) #right


wall5 = set_wall(field_margin / 2, field_margin / 2, field_size[0], field_size[1], False) #top
wall6 = set_wall(field_margin / 2, field_margin / 2 + field_size[1], field_size[0], field_size[1], False)#bottom

wall7 = set_wall(0, 0, window_size[0], 0, False)
wall8 = set_wall(0, 0, 0, window_size[1], True)

wall9 = set_wall(window_size[0], 0, window_size[0], window_size[1], True)
wall10 = set_wall(0, window_size[1], window_size[0], window_size[1], False)