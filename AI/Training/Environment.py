import torch
from Core.Objects.Map import Map
import Constants
from AI.Translator import Translator
from Player_actions.Net_action import Net_action

class Environment:
    def __init__(self, nn1, nn2, num_steps = 1024):
        self.num_steps = num_steps
        self.nn1 = nn1
        self.nn2 = nn2
        self.map = Map()

        #TODO исправить перебор игроков


        self.translator_team1 = Translator(self.map, nn1, self.map.players_team1[0], self.map.players_team2)
        self.translator_team2 = Translator(self.map, nn2, self.map.players_team2[0], self.map.players_team1)

        self.count = 0

        self.net_action_team1 = Net_action(self.map, nn1, self.map.players_team1[0])
        self.net_action_team2 = Net_action(self.map, nn2, self.map.players_team2[0])

        self.kick_count_modifier_team1 = 1
        self.kick_count_modifier_team2 = 1

    def step(self, action1, action2):

        self.net_action_team1.act()
        self.net_action_team2.act()
    
        self.map.move_balls()

        self.count += 1

        r1, r2, done = self.calculate_rewards()

        if not done:
            done = self.count > self.num_steps
        
        s1 = self.translator_team1.translate_input()
        s2 = self.translator_team2.translate_input()

        return (s1, s2), (r1, r2), done

    def reset(self):
        self.count = 0

        self.kick_count_modifier_team1 = 1
        self.kick_count_modifier_team2 = 1

        self.map.load()

        s1 = self.translator_team1.translate_input()
        s2 = self.translator_team1.translate_input()

        return (s1, s2)
    
    def calculate_rewards(self):
        r_team1 = 0
        r_team2 = 0

        if self.map.kick_flag_team1:
            r_team1 += 2000 * self.kick_count_modifier_team1

        if self.map.kick_flag_team2:
            r_team2 += 2000 * self.kick_count_modifier_team2
        

        player1 = self.map.players_team1[0]
        player2 = self.map.players_team2[0]
        ball = self.map.balls[0]


        d1 = player1.position - ball.position
        dist1 = torch.abs(d1.norm() - player1.radius - ball.radius + 0.00001)

        reward_big_dist1 = dist1
        
        d2 = player1.position - ball.position
        dist2 = torch.abs(d2.norm() - player2.radius - ball.radius + 0.00001)

        reward_big_dist2 = dist2
        
        r_team1 -= reward_big_dist1 / Constants.window_size[0]
        r_team2 -= reward_big_dist2 / Constants.window_size[0]

        done = False
        if player1.position[0] < 0 or player1.position[0] > Constants.window_size[0] or \
                player1.position[1] < 0 or player1.position[1] > Constants.window_size[1]:
            done = True
            r_team1 = -5000
        
        if player2.position[0] < 0 or player2.position[0] > Constants.window_size[0] or \
                player2.position[1] < 0 or player2.position[1] > Constants.window_size[1]:
            done = True
            r_team2 = -5000

        r_team1 = torch.tensor([r_team1])
        r_team2 = torch.tensor([r_team2])

        return r_team1, r_team2, done        
                    
                