from random import random

import torch
from Objects.Map import Map
from Objects.Gate import Gate
from Data_structure.Gates_data import Gates_data
from Physics.WallCollision import WallCollision as Wall
import Constants
from Player_actions import Net_action
import copy

class Environment:
    def __init__(self, map, nn):
        # Initialize Pygame
        self.num_steps = 2048
        self.map = Map(Constants.field_size[0], Constants.field_size[1])

        self.map.add_players()
        self.map.add_balls()

        # teams = [self.map.balls]
        teams = []
        
        self.teams_number = 1 # DO NOT TOUCH IT

        for i in range(self.teams_number):
            teams.append(copy.deepcopy(self.map.balls))

        self.map.set_ball_teams(teams)

        self.add_gates()
        self.add_walls()

        self.map.save_state()
        self.map = map
        self.ai_action = Net_action.Net_action(nn, map, 0)

        self.count = 0

        player = self.map.ball_teams[0][0]
        ball = self.map.ball_teams[0][1]

        d = player.position - ball.position
        dist = torch.abs(d.norm() - player.radius - ball.radius)

        self.prev_dist = dist
        self.k_dist = 1
        self.kick_count = 1
        
                

    def add_gates(self):
        gates_data = [Gates_data(is_left=True), Gates_data(is_left=False)]
        for data in gates_data:
            gate = Gate(data)
            gate.add_boundaries()
            self.map.add_gate(gate)

    def add_walls(self):
        walls = [Constants.wall1, Constants.wall2, Constants.wall3, Constants.wall4, Constants.wall5, 
                 Constants.wall6]
        for wall in walls:
            self.map.add_wall(Wall(wall.start, wall.end, wall.constant, wall.is_vertical))

    def step(self, input):

            self.ai_action.act(input)    
            self.map.move_balls()

            done = self.count > self.num_steps
            self.count += 1

            be_strict = True
            r = 0
            if self.map.kick_flag:
                r += 2000 * self.kick_count
                self.map.kick_flag = False
                be_strict = False
                # done = True

            player = self.map.ball_teams[0][0]
            ball = self.map.ball_teams[0][1]
            d = player.position - ball.position

            
            dist = torch.abs(d.norm() - player.radius - ball.radius + 0.00001)

            reward_big_dist = dist

            self.k_dist = self.k_dist ** (1 - self.k_dist)

            if not be_strict:
                 self.k_dist = 0.7
                

            if self.num_steps - self.count <= -3:
                print("error in env.count")

            r = r - 1 * reward_big_dist / Constants.field_size[0] / (self.num_steps - self.count + 3)
            self.prev_dist = dist
            
            ns = self.ai_action.translator.translate_input()
            return ns, r, done

    def reset(self):
        # with torch.no_grad():
            self.count = 0

            # if not self.flag:
            self.map.load()
            player = self.map.ball_teams[0][0]
            ball = self.map.ball_teams[0][1]

            d = player.position - ball.position

            dist = torch.abs(d.norm() - player.radius - ball.radius)

            self.prev_dist = dist

            self.k_dist = 0.7 + 1 / dist


            self.step_num = dist * 1 * Constants.max_ball_speed / (Constants.speed_increment)

            self.map.ball_teams[0][0].velocity.x = torch.tensor(0.).to(Constants.device)
            self.map.ball_teams[0][0].velocity.y = torch.tensor(0.).to(Constants.device)
            self.map.ball_teams[0][1].velocity.x = torch.tensor(0.).to(Constants.device)
            self.map.ball_teams[0][1].velocity.y = torch.tensor(0.).to(Constants.device)
            

            self.ai_action.set_player(self.map.ball_teams[0][0])
            inp = self.ai_action.translator.translate_input()

            return inp
        
                    
                