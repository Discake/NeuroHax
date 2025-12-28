from Core.Domain.GameConfig import BallsConfig, FieldConfig, GameConfig, GateConfig, MapConfig, PhysicsConfig, TeamConfig, WallConfig

map_width = 1000
map_height = 800

map = MapConfig(map_width, map_height)

field_width = 800
field_height = 600
field_x_center = 500
field_y_center = 400

field = FieldConfig(field_width, field_height, field_x_center, field_y_center)

gate_width = 50
gate_height = 200

right_gate_x_center = field.x_center + field.width / 2 + gate_width / 2
left_gate_x_center = field.x_center - field.width / 2 - gate_width / 2

gate_y_center = field.y_center

gates = [GateConfig(gate_width, gate_height, left_gate_x_center, gate_y_center),
                  GateConfig(gate_width, gate_height, right_gate_x_center, gate_y_center)]

physics_tick_rate = 60
physics_ball_radius = 10
physics_player_radius = 20
physics_ball_mass = 0.5
physics_player_mass = 1
physics_kick_radius = 15
physics_kick_power = 3
physics_friction = 0.3
physics_player_max_speed = 7
physics_ball_max_speed = 30
physics_player_move_modificator = 0.1

physics = PhysicsConfig(physics_tick_rate, physics_ball_radius, physics_player_radius,
                         physics_ball_mass, physics_player_mass, physics_kick_radius,
                           physics_kick_power, physics_friction, physics_player_max_speed,
                             physics_ball_max_speed, physics_player_move_modificator)

left_gate_wall1 = WallConfig((left_gate_x_center - gate_width / 2, gate_y_center - gate_height / 2),
                            (left_gate_x_center - gate_width / 2, gate_y_center + gate_height / 2))
left_gate_wall2 = WallConfig((left_gate_x_center - gate_width / 2, gate_y_center + gate_height / 2),
                            (left_gate_x_center + gate_width / 2, gate_y_center + gate_height / 2))
left_gate_wall3 = WallConfig((left_gate_x_center - gate_width / 2, gate_y_center - gate_height / 2),
                            (left_gate_x_center + gate_width / 2, gate_y_center - gate_height / 2))
right_gate_wall1 = WallConfig((right_gate_x_center + gate_width / 2, gate_y_center - gate_height / 2),
                            (right_gate_x_center + gate_width / 2, gate_y_center + gate_height / 2))
right_gate_wall2 = WallConfig((right_gate_x_center - gate_width / 2, gate_y_center + gate_height / 2),
                            (right_gate_x_center + gate_width / 2, gate_y_center + gate_height / 2))
right_gate_wall3 = WallConfig((right_gate_x_center - gate_width / 2, gate_y_center - gate_height / 2),
                            (right_gate_x_center + gate_width / 2, gate_y_center - gate_height / 2))

bottom_wall = WallConfig((field.x_center - field.width / 2, field.y_center - field.height / 2),
                         (field.x_center + field.width / 2, field.y_center - field.height / 2))
top_wall = WallConfig((field.x_center - field.width / 2, field.y_center + field.height / 2),
                      (field.x_center + field.width / 2, field.y_center + field.height / 2))
left_wall1 = WallConfig((field.x_center - field.width / 2, field.y_center - field.height / 2),
                        (field.x_center - field.width / 2, gate_y_center - gate_height / 2))
left_wall2 = WallConfig((field.x_center - field.width / 2, gate_y_center + gate_height / 2),
                        (field.x_center - field.width / 2, field.y_center + field.height / 2))
right_wall1 = WallConfig((field.x_center + field.width / 2, field.y_center - field.height / 2),
                        (field.x_center + field.width / 2, gate_y_center - gate_height /2 ))
right_wall2 = WallConfig((field.x_center + field.width / 2, gate_y_center + gate_height / 2),
                        (field.x_center + field.width / 2, field.y_center + field.height / 2))

general_walls = [left_gate_wall1, left_gate_wall2, left_gate_wall3, right_gate_wall1, right_gate_wall2, right_gate_wall3]
ball_walls = [bottom_wall, top_wall, left_wall1, left_wall2, right_wall1, right_wall2]

balls_max_balls = 1
balls_balls_positions = [(field.x_center, field.y_center)]

balls = BallsConfig(balls_max_balls, balls_balls_positions)

team_blue_max_players = 1
team_blue_players_positions = [(field.x_center + 100, field.y_center)]
team_blue = TeamConfig(team_blue_max_players, team_blue_players_positions)

team_red_max_players = 1
team_red_players_positions = [(field.x_center - 100, field.y_center)]
team_red = TeamConfig(team_red_max_players, team_red_players_positions)

game_config = GameConfig(map, field, gates, physics, team_red, team_blue, balls, general_walls, ball_walls)

