import torch
from Core.Objects.Map import Map
import Constants
from Player_actions.Net_action import Net_action

class Environment:
    def __init__(self, nn1, nn2, num_steps = 1024):
        self.num_steps = num_steps
        self.nn1 = nn1
        self.nn2 = nn2
        self.map = Map()

        #TODO исправить перебор игроков

        self.count = 0

        self.net_action_team1 = Net_action(self.map, nn1, self.map.players_team1[0], is_team_1=True)
        self.net_action_team2 = Net_action(self.map, nn2, self.map.players_team2[0], is_team_1=False)

        self.kick_count_modifier_team1 = 1
        self.kick_count_modifier_team2 = 1

    def step(self, action1, action2):

        self.net_action_team1.act(action1)
        self.net_action_team2.act(action2)
    
        self.map.move_balls()

        self.count += 1

        r1, r2, done = self.calculate_rewards()

        if not done:
            done = self.count > (self.num_steps - 3)

        if done:
            print("DONE!")
        
        s1 = self.net_action_team1.translator.translate_input()
        s2 = self.net_action_team2.translator.translate_input()

        return (s1, s2), (r1, r2), done

    def reset(self):
        self.count = 0

        self.kick_count_modifier_team1 = 1
        self.kick_count_modifier_team2 = 1

        self.map.load()

        player1 = self.map.players_team1[0]
        player2 = self.map.players_team2[0]
        ball = self.map.balls[0]

        d1 = player1.position - ball.position
        sqrt1 = torch.linalg.vector_norm(d1)
        dist1 = max(0,sqrt1 - player1.radius - ball.radius)
        
        d2 = player2.position - ball.position
        sqrt2 = torch.linalg.vector_norm(d2)
        dist2 = max(0,sqrt2 - player2.radius - ball.radius)

        self.prev_dist1 = dist1
        self.prev_dist2 = dist2

        s1 = self.net_action_team1.translator.translate_input()
        s2 = self.net_action_team2.translator.translate_input()

        return (s1, s2)
    
    def calculate_rewards(self):
        r_team1 = 0
        r_team2 = 0

        if self.map.kick_flag_team1:
            r_team1 += 200
            self.kick_count_modifier_team1 += 1

        if self.map.kick_flag_team2:
            r_team2 += 200
            self.kick_count_modifier_team2 += 1
        

        player1 = self.map.players_team1[0]
        player2 = self.map.players_team2[0]
        ball = self.map.balls[0]

        
        d1 = player1.position - ball.position
        sqrt1 = torch.linalg.vector_norm(d1)
        dist1 = max(0,sqrt1 - player1.radius - ball.radius)

        reward_big_dist1 = dist1
        
        d2 = player2.position - ball.position
        sqrt2 = torch.linalg.vector_norm(d2)
        dist2 = max(0,sqrt2 - player2.radius - ball.radius)

        reward_big_dist2 = dist2
        
        r_team1 -= reward_big_dist1 * 2 / Constants.window_size[0]
        r_team2 -= reward_big_dist2 * 2 / Constants.window_size[0]

        r_team1 += (self.prev_dist1 - dist1)
        r_team2 += (self.prev_dist2 - dist2)

        self.prev_dist1 = dist1
        self.prev_dist2 = dist2

        # r_team1 -= reward_big_dist1
        # r_team2 -= reward_big_dist2

        done = False
        # if player1.position[0] < Constants.field_margin / 2 or player1.position[0] > Constants.window_size[0] - Constants.field_margin / 2 or \
        #         player1.position[1] < Constants.field_margin / 2 or player1.position[1] > Constants.window_size[1] - Constants.field_margin / 2:
        #     r_team1 = -5000
        
        # if player2.position[0] < Constants.field_margin / 2 or player2.position[0] > Constants.window_size[0] - Constants.field_margin / 2 or \
        #         player2.position[1] < Constants.field_margin / 2 or player2.position[1] > Constants.window_size[1] - Constants.field_margin / 2:
        #     r_team2 = -5000


        if player1.position[0] < 0 or player1.position[0] > Constants.window_size[0] or \
                player1.position[1] < 0 or player1.position[1] > Constants.window_size[1]:
            done = True
            r_team1 = -500 #* (self.num_steps - self.count + 3)
        
        if player2.position[0] < 0 or player2.position[0] > Constants.window_size[0] or \
                player2.position[1] < 0 or player2.position[1] > Constants.window_size[1]:
            done = True
            r_team2 = -500 #* (self.num_steps - self.count + 3)

        r_team1 = torch.tensor([r_team1])
        r_team2 = torch.tensor([r_team2])

        return r_team1, r_team2, done        
                    
                