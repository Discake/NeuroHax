import torch
from Objects.Ball import Ball
from Objects.Player import Player
import Constants
import random

from Physics.WallCollision import WallCollision

class Map:
    def __init__(self):
        self.balls = []

        self.players_team1 = []
        self.players_team2 = []
        self.gates = []
        self.walls = []
        self.score = [0, 0]
        self.kick_flag = False

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
            self.balls.append(player)

    def move_balls(self):
        all_balls = self.balls + self.players_team1 + self.players_team2

        # Итерации нужны для обеспечения разрешимости коллизий при высоких скоростях
        for _ in range(Constants.iterations):
            for ball in all_balls:
                ball.move(Constants.time_increment / Constants.iterations)

            for i in range(len(all_balls)):
                ball = all_balls[i]
                ball.is_collided = False

                # Коллизия шаров друг с другом
                for j in range(i + 1, len(all_balls)):
                    other_ball = all_balls[j]
                    if ball.detect_collision(other_ball):
                        ball.resolve_collision(other_ball)

                # Коллизия шаров и ворот
                for gate in self.gates:
                    for line in gate.boundaries:
                        if line.detect_collision(ball):
                            line.resolve_collision(ball)

                # Пинок мяча
                if isinstance(ball, Player):
                    for other_ball in self.balls:
                        if not ball.is_kicking:
                            self.kick(ball, other_ball) 
            
            # Коллизия мяча и границы поля
            for ball in self.balls:
                for wall in self.walls:
                    if wall.detect_collision(ball):
                        wall.resolve_collision(ball)
                        self.wall_hit = True          


    def kick(self, ball, other_ball):
        ball_pos = ball.position
        other_ball_pos = other_ball.position
                                
        direction = ball_pos - other_ball_pos
        dist = torch.dot(direction, direction)
        if dist < (ball.radius + other_ball.radius + Constants.kick_radius) ** 2:
            ball.kick(other_ball, direction)
            self.kick_flag = True

    def load_random(self): # Метод загрузки первого игрока на поле в случайной позиции

        self.kick_flag = False
        self.hit_flag = False
        
        is_right = random.randint(0, 1)
        y_pos = random.randint(0, Constants.field_size[1] - self.players_team1[0].radius)

        if is_right:
            self.players_team1.position[0] = Constants.players_positions_team1[0][0]
        else:
            self.players_team1[0].position[0] = Constants.players_positions_team2[0][0]
        self.players_team1[0].position[1] = Constants.field_margin / 2 + y_pos

        self.balls[0].position = torch.tensor([Constants.x_center, Constants.y_center], device=Constants.device)

    def load_random(self): # Метод загрузки всех игроков и мячей на начальные позиции
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
    new_map = Map(old_map.width, old_map.height)
    new_map.score = list(old_map.score)
    new_map.save = old_map.save
    new_map.hit_flag = old_map.hit_flag
    new_map.kick_flag = old_map.kick_flag
    new_map.wall_hit = old_map.wall_hit
    new_map.ball_teams = [[clone_and_detach_object(ball) for ball in team] for team in old_map.ball_teams]
    new_map.balls = [clone_and_detach_object(ball) for ball in old_map.balls]
    new_map.players = [clone_and_detach_object(player) for player in old_map.players]
    new_map.gates   = [clone_and_detach_object(gate) for gate in old_map.gates]
    new_map.walls   = [clone_and_detach_object(wall) for wall in old_map.walls]
    return new_map
    
    