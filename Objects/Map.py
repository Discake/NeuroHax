import torch
from Objects.Ball import Ball
from Objects.Player import Player
import Constants
import numpy as np
import copy
import random

class Map:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.balls = []
        self.players = []
        self.gates = []
        self.walls = []
        self.score = [0, 0]
        self.save = None
        self.hit_flag = False
        self.kick_flag = False
        self.wall_hit = False

    def set_field(self, field):
        self.field = field

    def add_gate(self, gate):
        self.gates.append(gate)

    def add_wall(self, wall):
        self.walls.append(wall)

    
    def add_players(self):
        for i in range(Constants.player_number):
            player1 = Player(
                Constants.players_positions_team1[i],
                Constants.player_radius,
                Constants.player_mass,
                Constants.max_player_speed)
            player1.set_color(Constants.player_color)
            self.players.append(player1)
            self.balls.append(player1)

            # player2 = Player(
            #     Constants.players_positions_team2[i],
            #     Constants.player_radius,
            #     Constants.player_mass,
            #     Constants.max_player_speed)
            # player2.set_color(Constants.player_color)
            # self.players.append(player2)
            # self.balls.append(player2)

    def set_ball_teams(self, ball_teams):
        self.ball_teams = ball_teams

    def move_balls(self):
        
        for _ in range(Constants.iterations):
            for team in self.ball_teams:
                for ball in team:
                    ball.is_collided = False
                    for other_ball in team:
                        if ball != other_ball and not ball.is_collided :
                            if ball.detect_collision(other_ball):
                                ball.resolve_collision(other_ball)
                                ball.is_collided = True
                                if(isinstance(ball, Player) or isinstance(other_ball, Player)):
                                    self.hit_flag = True
                                    self.hit_player = other_ball

                    ball.move(Constants.time_increment / Constants.iterations)

                    for gate in self.gates:
                        for line in gate.boundaries:
                            line.resolve_collision(ball)
                    for wall in self.walls:
                        if wall.detect_collision:
                            wall.resolve_collision(ball)
                            self.wall_hit = True


                    if isinstance(ball, Player):
                        for other_ball in team:
                            if not isinstance(other_ball, Player) and ball.is_kicking:
                                if isinstance(ball.position.x, torch.Tensor):
                                    ball_pos = torch.stack([ball.position.x, ball.position.y])
                                else:
                                    ball_pos = torch.stack([ball.position.x, ball.position.y])

                                if isinstance(other_ball.position.x, torch.Tensor):
                                    other_ball_pos = torch.stack([other_ball.position.x, other_ball.position.y])
                                else:
                                    other_ball_pos = torch.stack([other_ball.position.x, other_ball.position.y])
                                
                                
                                direction = ball_pos - other_ball_pos
                                dist = torch.linalg.vector_norm(direction)
                                if dist < ball.radius + other_ball.radius + Constants.kick_radius:
                                    ball.kick(other_ball, direction)
                                    self.kick_flag = True


            
            # if not isinstance(ball, Player):
            #     if ball.position.x <= Constants.left_gates_center_x + Constants.gates_outer_width:
            #         self.score[1] += 1
            #         self.load()
            #         self.save_state()
            #     if ball.position.x >= Constants.right_gates_center_x:
            #         self.score[0] += 1
            #         self.load()
            #         self.save_state()
            

    def save_state(self):
        # self.save = copy.deepcopy(self.ball_teams)

        # for team in self.ball_teams:
        #     for ball in team:
        #         print("Saved: ")
        #         print(f"{ball}")
        pass

    def load(self):

        self.kick_flag = False
        self.hit_flag = False
        # self.ball_teams.clear()

        # for _ in range(num):
        #     self.ball_teams.append(copy.deepcopy(self.balls))
        
        # self.ball_teams = self.save
        is_right = random.randint(0, 1)
        y_pos = random.randint(0, Constants.field_size[1] - self.ball_teams[0][0].radius)



        if is_right:
            self.ball_teams[0][0].position.x = Constants.players_positions_team1[0].x
        else:
            self.ball_teams[0][0].position.x = Constants.players_positions_team2[0].x
        self.ball_teams[0][0].position.y = torch.tensor(Constants.field_margin / 2 + y_pos).to(Constants.device)

        self.ball_teams[0][1].position.x = torch.tensor(Constants.x_center).to(Constants.device)
        self.ball_teams[0][1].position.y = torch.tensor(Constants.y_center).to(Constants.device)
        # self.players.clear()
        # for team in self.ball_teams:
        #     for i in range(len(team) - 1):
        #         self.players.append(team[i])
    
    def add_balls(self):
        for i in range(Constants.ball_number):
            ball = Ball(Constants.balls_positions[i], Constants.ball_radius, Constants.ball_mass, Constants.max_ball_speed)
            ball.set_color(Constants.ball_color)
            
            self.balls.append(ball)
    
    