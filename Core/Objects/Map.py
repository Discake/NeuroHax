import torch
from Core.Data_structure.Gates_data import Gates_data
from Core.Objects.Ball import Ball
from Core.Objects.Gate import Gate
from Core.Objects.Player import Player
import Constants
import random

from Core.Physics.WallCollision import WallCollision

class Map:
    def __init__(self):
        self.balls = list[Ball]()

        self.players_team1 = list[Player]()
        self.players_team2 = list[Player]()
        self.gates = list[Gate]()
        self.walls = list[WallCollision]()

        self.score = [0, 0]
        self.kick_flag_team1 = False
        self.kick_flag_team2 = False
        self.time_increment = Constants.time_increment

        self.add_balls()
        self.add_players()
        self.add_gates()
        self.add_walls()

        self.all_balls = list[Ball]()
        self.all_balls = self.balls + self.players_team1 + self.players_team2
    
    def add_gates(self):
        gates_data = [Gates_data(is_left=True), Gates_data(is_left=False)]
        for data in gates_data:
            gate = Gate(data)
            gate.add_boundaries()
            self.add_gate(gate)

    def add_walls(self):
        walls = [Constants.wall1, Constants.wall2, Constants.wall3, Constants.wall4, Constants.wall5, 
                 Constants.wall6]
        for wall in walls:
            self.add_wall(WallCollision(wall.start, wall.end, wall.constant, wall.is_vertical))

    def set_field(self, field):
        self.field = field

    def add_gate(self, gate):
        self.gates.append(gate)

    def add_wall(self, wall : WallCollision):
        self.walls.append(wall)

    
    def add_players(self):
        for i in range(Constants.player_number):
            player = Player(
                Constants.players_positions_team1[i],
                Constants.player_radius,
                Constants.player_mass,
                Constants.max_player_speed)
            player.set_color(Constants.player_color)
            self.players_team1.append(player)

        for i in range(Constants.player_number):
            player = Player(
                Constants.players_positions_team2[i],
                Constants.player_radius,
                Constants.player_mass,
                Constants.max_player_speed)
            player.set_color(Constants.player_color)
            self.players_team2.append(player)

    def move_balls(self):
        
        self.kick_flag_team1 = False
        self.kick_flag_team2 = False
        # Итерации нужны для обеспечения разрешимости коллизий при высоких скоростях
        for _ in range(Constants.iterations):
            for ball in self.all_balls:
                    ball.move(self.time_increment / Constants.iterations)
            
            for i in range(len(self.all_balls)):
                ball = self.all_balls[i]

                # Коллизия шаров друг с другом
                for j in range(i + 1, len(self.all_balls)):
                    other_ball = self.all_balls[j]
                    if ball.detect_collision(other_ball):
                        ball.resolve_collision(other_ball)

                # Коллизия шаров и ворот
                for gate in self.gates:
                    for line in gate.boundaries:
                        if line.detect_collision_pure_python(ball):
                            line.resolve_collision(ball)
            
            # Коллизия мяча и границы поля
            for ball in self.balls:
                for wall in self.walls:
                    if wall.detect_collision_pure_python(ball):
                        wall.resolve_collision(ball)
                        self.wall_hit = True

            # Пинок мяча
            for player in self.players_team1:
                for ball in self.balls:
                    if player.is_kicking:
                        self.kick(player, ball, is_team_1=True)
                        

            for player in self.players_team2:
                for ball in self.balls:
                    if player.is_kicking:
                        self.kick(player, ball, is_team_1=False)


    def kick(self, ball : Player, other_ball : Ball, is_team_1):
        ball_pos = ball.position
        other_ball_pos = other_ball.position

        if abs(ball_pos[0] - other_ball_pos[0]) >  ball.radius + other_ball.radius + Constants.kick_radius or \
                abs(ball_pos[1] - other_ball_pos[1]) > ball.radius + other_ball.radius + Constants.kick_radius:
            return
                                
        direction = ball_pos - other_ball_pos
        dist = torch.dot(direction, direction)
        if dist < (ball.radius + other_ball.radius + Constants.kick_radius) ** 2:
            ball.kick(other_ball, direction)
            self.kick_flag = True
            if is_team_1:
                self.kick_flag_team1 = True
            else:
                self.kick_flag_team2 = True

    def load_random(self): # Метод загрузки первого игрока на поле в случайной позиции

        self.kick_flag = False
        self.hit_flag = False
        self.time_increment = (random.random() - 0.5) * Constants.time_increment + Constants.time_increment 
        
        is_right = random.randint(0, 1)
        y_pos = random.randint(0, Constants.field_size[1] - self.players_team1[0].radius)

        if is_right:
            self.players_team1.position[0] = Constants.players_positions_team1[0][0]
        else:
            self.players_team1[0].position[0] = Constants.players_positions_team2[0][0]
        self.players_team1[0].position[1] = Constants.field_margin / 2 + y_pos

        self.balls[0].position = torch.tensor([Constants.x_center, Constants.y_center], device=Constants.device)

    def load(self): # Метод загрузки всех игроков и мячей на начальные позиции
        self.kick_flag = False

        for i, player in enumerate(self.players_team1):
            player.position = Constants.players_positions_team1[i]
            player.velocity = torch.tensor([0., 0.], device=Constants.device)

        for i, player in enumerate(self.players_team2):
            player.position = Constants.players_positions_team2[i]
            player.velocity = torch.tensor([0., 0.], device=Constants.device)

        for i, ball in enumerate(self.balls):
            ball.position = Constants.balls_positions[i]
            ball.velocity = torch.tensor([0., 0.], device=Constants.device)     
    
    def add_balls(self):
        for i in range(Constants.ball_number):
            ball = Ball(Constants.balls_positions[i], Constants.ball_radius, Constants.ball_mass, Constants.max_ball_speed)
            ball.set_color(Constants.ball_color)
            
            self.balls.append(ball)

# метод клонирования объектов карты
def clone_and_detach_object(obj):
    if hasattr(obj, '__dict__'):
        cloned_obj = obj.__class__.__new__(obj.__class__)
        for k, v in obj.__dict__.items():
            if isinstance(v, torch.Tensor):
                # detach + clone (на CPU - detach() делает копию без градиентов)
                setattr(cloned_obj, k, v.detach().clone())
            elif isinstance(v, list):
                setattr(cloned_obj, k, [clone_and_detach_object(x) if hasattr(x, '__dict__') else x for x in v])
            elif isinstance(v, dict):
                setattr(cloned_obj, k, {kk: clone_and_detach_object(vv) for kk, vv in v.items()})
            else:
                setattr(cloned_obj, k, v)
        return cloned_obj
    else:
        return obj

# Метод клонирования карты
def clone_and_detach_map(old_map):
    new_map = Map()
    new_map.score = list(old_map.score)
    new_map.kick_flag = old_map.kick_flag
    new_map.players_team1 = [clone_and_detach_object(player) for player in old_map.players_team1]
    new_map.players_team2 = [clone_and_detach_object(player) for player in old_map.players_team2]
    new_map.balls = [clone_and_detach_object(ball) for ball in old_map.balls]
    new_map.gates   = [clone_and_detach_object(gate) for gate in old_map.gates]
    new_map.walls   = [clone_and_detach_object(wall) for wall in old_map.walls]
    return new_map
    
    