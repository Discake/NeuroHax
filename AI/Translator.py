import torch
from AI.Maksigma_net import Maksigma_net
import Constants
from Core.Objects.Map import Map

class Translator:
    def __init__(self, map: Map, net : Maksigma_net):
        self.map = map
        self.net = net
        self.input = torch.zeros(Constants.player_number * 4 + Constants.ball_number * 4 + 1)
        self.output = [0, 0, 0]

    def translate_input(self):
        input1 = torch.zeros(4)
        # input2 = torch.zeros(4)
        input3 = torch.zeros(4)
        maximum = Constants.field_size[1]

        
        # for i in range(len(self.map.players_team1)):
        for i in range(1):
            player = self.map.players_team1[i]
            
            input1[4 * i] = (player.position[0] - Constants.x_center) / (maximum)
            input1[4 * i + 1] = (player.position[1] - Constants.y_center) / (maximum)
            input1[4 * i + 2] = (player.velocity[0]) / Constants.max_player_speed / 2
            input1[4 * i + 3] = (player.velocity[1]) / Constants.max_player_speed / 2

        # for i in range(len(self.map.players_team2)):
        #     player = self.map.players_team2[i]
            
        #     input2[4 * i] = (player.position[0] - Constants.x_center) / (maximum)
        #     input2[4 * i + 1] = (player.position[1] - Constants.y_center) / (maximum)
        #     input2[4 * i + 2] = (player.velocity[0]) / Constants.max_player_speed / 2
        #     input2[4 * i + 3] = (player.velocity[1]) / Constants.max_player_speed / 2
        
        for i in range(len(self.map.balls)):
            ball = self.map.balls[i]
            
            input3[4 * i] = (ball.position[0] - Constants.x_center) / (maximum)
            input3[4 * i + 1] = (ball.position[1] - Constants.y_center) / (maximum)
            input3[4 * i + 2] = (ball.velocity[0]) / Constants.max_ball_speed / 2
            input3[4 * i + 3] = (ball.velocity[1]) / Constants.max_ball_speed / 2
                
        input4 = torch.tensor([self.map.time_increment / (Constants.time_increment * 5)], device=Constants.device)


        input = torch.cat((input1, input3, input4))
            

        for item in input:
            if item > 1 or item < -1:
                print("Error in input")

        return input