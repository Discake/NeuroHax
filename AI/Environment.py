import torch
from Objects.Map import Map
from Objects.Gate import Gate
from Data_structure.Gates_data import Gates_data
from Physics.WallCollision import WallCollision as Wall
import Constants
from Player_actions import Net_action

class Environment:
    def __init__(self, nn, num_steps = 1024):
        self.num_steps = 1024
        self.map = Map()

        self.ai_action = Net_action.Net_action(nn, self.map)

        self.count = 0

        player = self.map.players_team1[0]
        ball = self.map.balls[0]

        d = player.position - ball.position
        dist = torch.abs(d.norm() - player.radius - ball.radius)

        self.prev_dist = dist
        self.kick_count_modifier = 1

    def step(self, input):

            self.ai_action.act(input)    
            self.map.move_balls()

            done = self.count > self.num_steps
            self.count += 1

            r = 0
            if self.map.kick_flag:
                r += 2000 * self.kick_count_modifier
                self.map.kick_flag = False
                # done = True

            player = self.map.players_team1[0]
            ball = self.map.balls[0]
            d = player.position - ball.position

            
            dist = torch.abs(d.norm() - player.radius - ball.radius + 0.00001)

            reward_big_dist = dist
                

            if self.num_steps - self.count <= -3:
                print("error in env.count")

            if player.position[0] < 0 or player.position[0] > Constants.window_size[0] or \
                  player.position[1] < 0 or player.position[1] > Constants.window_size[1]:
                done = True
                r = r - 1000

            r = r - 1 * reward_big_dist / Constants.field_size[0] / (self.num_steps - self.count + 3)
            self.prev_dist = dist
            
            ns = self.ai_action.translator.translate_input()
            return ns, r, done

    def reset(self):
        self.count = 0

        self.map.load_random()
        player = self.map.players_team1[0]
        ball = self.map.balls[0]

        d = player.position - ball.position

        dist = torch.abs(d.norm() - player.radius - ball.radius)

        self.prev_dist = dist            

        self.ai_action.set_player(self.map.players_team1[0])
        inp = self.ai_action.translator.translate_input()

        return inp
        
                    
                