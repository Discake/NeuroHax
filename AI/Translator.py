import torch
from AI.Maksigma_net import Maksigma_net
import Constants
from Objects.Map import Map
from Objects.Player import Player

class Translator:
    def __init__(self, map: Map, net : Maksigma_net, index):
        self.map = map
        self.net = net
        self.input = torch.zeros(Constants.player_number * 4 + Constants.ball_number * 4 + 1)
        self.output = [0, 0, 0]
        self.index = index

    def translate_input(self):
        # ball = self.map.balls[-1]

        # self.input = self.input.detach().clone()
        input = [0, 0, 0, 0, 0, 0, 0, 0]
        maximum = Constants.field_size[1]
        
        for i in range(len(self.map.ball_teams[self.index])):
            ball = self.map.ball_teams[self.index][i]

            if isinstance(ball, Player):
                input[4 * i] = (ball.position[0] - Constants.x_center) / (maximum)
                input[4 * i + 1] = (ball.position[1] - Constants.y_center) / (maximum)
                input[4 * i + 2] = (ball.velocity[0]) / Constants.max_player_speed / 2
                input[4 * i + 3] = (ball.velocity[1]) / Constants.max_player_speed / 2
            else:
                input[4 * i] = (ball.position[0] - Constants.x_center) / (maximum)
                input[4 * i + 1] = (ball.position[1] - Constants.y_center) / (maximum)
                input[4 * i + 2] = (ball.velocity[0]) / Constants.max_ball_speed / 2
                input[4 * i + 3] = (ball.velocity[1]) / Constants.max_ball_speed / 2
                

        input = torch.stack(input)
            

        for item in input:
            if item > 1 or item < -1:
                print("Error in input")

        return input


    def translate_output(self, input):
        
        action, logp, _, entropy = self.net.get_action(input)

        return action, logp, entropy
        

        
            
                

