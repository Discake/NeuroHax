from random import random

import torch
from Objects.Map import Map
from Objects.Gate import Gate
from Data_structure.Gates_data import Gates_data
from Physics.WallCollision import WallCollision as Wall
import Constants
from Player_actions import Net_action
import copy

class Enviroment:
    def __init__(self, map, nn):
        # Initialize Pygame

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
        self.r = 5000

        player = self.map.ball_teams[0][0]
        ball = self.map.ball_teams[0][1]

        dx = torch.abs(player.position.x - ball.position.x)
        dy = torch.abs(player.position.y - ball.position.y)
        dist = torch.abs(torch.sqrt(dx ** 2 + dy ** 2) - player.radius - ball.radius)

        self.prev_dist = dist
        self.step_num = 0
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
                 Constants.wall6, Constants.wall7, Constants.wall8, Constants.wall9, Constants.wall10]
        for wall in walls:
            self.map.add_wall(Wall(wall.start, wall.end, wall.constant, wall.is_vertical))

    def step(self, input):
        # self.ai_action.set_player(self.map.ball_teams[0][0])
        # self.ai_action.update_translator()

            self.ai_action.act(input)
        # with torch.no_grad():    
            self.map.move_balls()

            self.count += 1
            done = self.count > 500

            # r = 0
            be_strict = True
            r = 0
            if self.map.kick_flag:
                r += 2000 * self.kick_count
                self.map.kick_flag = False
                be_strict = False
                # done = True
            elif self.map.hit_flag:
                r += 2000 * self.kick_count
                self.map.hit_flag = False
                be_strict = False
                # done = True

            # if self.map.wall_hit:
            #     r -= 1000
            #     self.map.wall_hit = False

            player = self.map.ball_teams[0][0]
            ball = self.map.ball_teams[0][1]
            dx = (player.position.x - ball.position.x)
            dy = (player.position.y - ball.position.y)
            
            dist = (torch.sqrt(dx ** 2 + dy ** 2) - player.radius - ball.radius)
            
            # r = r - torch.sqrt(player.velocity.x ** 2 + player.velocity.y ** 2) / Constants.max_player_speed
            # r = r + torch.sqrt(ball.velocity.x ** 2 + ball.velocity.y ** 2) / Constants.max_ball_speed

            reward_prev_dist_less = (self.prev_dist - dist) * 1000# * Constants.field_size[0] / (player.radius + ball.radius) / 2
            reward_big_dist = dist


            # if dist > Constants.field_size[1] / 2:
            #     r += 10 * reward_prev_dist_less
            # else:
            #     r -= 1 * reward_big_dist

            self.k_dist = self.k_dist ** (1 - self.k_dist)

            if not be_strict:
                 self.k_dist = 0.7
            
            r = r + 1 * reward_prev_dist_less/ Constants.field_size[0]
                
            r = r - 1 * reward_big_dist * self.k_dist / Constants.field_size[0]
            # r -= 100 * reward_big_dist * self.k / (self.step_num - self.count + 100)

            # be_strict = ball.velocity.length() < 0.5
            # if be_strict and tick > 0:
            #     be_strict = False
            #     tick -= 1

            # if be_strict:
            # if be_strict:
            #     r += 0.1 * reward_prev_dist_less# / (Constants.window_size[0] / 2)
            #     r -= 0.01 * reward_big_dist# / (Constants.window_size[0] / 2)
                # if self.prev_dist > dist:
                #     r += 1
                
            # if dist < 3 * (player.radius + ball.radius):
            #     r += 0.3
            # else:
            #     r -= 0.3


            # r += player.velocity.length() / 2 * Constants.max_player_speed

            # r -= (Constants.max_ball_speed - ball.velocity.length()) / Constants.max_ball_speed

            self.prev_dist = dist
            # if self.count % 100 == 0:
            # r += 0.1 / dist
            # if dist < 120:
            #     r += 0.25
            #     # r -= 10 / (dist)
            # if dist > 200:
            #     r -= 0.5

            # if dist < 300:
            #     r -= 0.25
            # else:
                # r += 1000 / dist
            
            # if dist < self.prev_dist:
            #     r += self.r
            #     self.r += 1000

            # if dist > self.prev_dist:
            #     self.r = 5000
            #     r =  0
            # self.prev_dist = dist  

            # if self.map.score[0] > 0:
            #     done = True
            #     r += 10000

            #     self.map.score[0] = 0
            # if self.map.score[1] > 0:
            #     done = True
            #     r += 10000
            #     self.map.score[1] = 0

            # player = self.map.ball_teams[0][0]
            # r = player.position.x

            ns = self.ai_action.translator.translate_input()
            return ns, r, done

    def reset(self):
        # with torch.no_grad():
            self.count = 0

            # if not self.flag:
            self.map.load()
            player = self.map.ball_teams[0][0]
            ball = self.map.ball_teams[0][1]

            dx = torch.abs(player.position.x - ball.position.x)
            dy = torch.abs(player.position.y - ball.position.y)
            dist = torch.abs(torch.sqrt(dx ** 2 + dy ** 2) - player.radius - ball.radius)

            self.prev_dist = dist

            self.k_dist = 0.5 + 1 / dist


            self.step_num = dist * 1 * Constants.max_ball_speed / (Constants.speed_increment)

            self.map.ball_teams[0][0].velocity.x = torch.tensor(0.).to(Constants.device)
            self.map.ball_teams[0][0].velocity.y = torch.tensor(0.).to(Constants.device)
            self.map.ball_teams[0][1].velocity.x = torch.tensor(0.).to(Constants.device)
            self.map.ball_teams[0][1].velocity.y = torch.tensor(0.).to(Constants.device)
            # for team in self.map.ball_teams:
            #     for ball in team:
            #         print(f"{ball}")

            self.ai_action.set_player(self.map.ball_teams[0][0])
            # self.ai_action.update_translator()
            inp = self.ai_action.translator.translate_input()

            return inp
        
                    
                