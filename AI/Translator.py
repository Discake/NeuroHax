import torch
from AI.Maksigma_net import Maksigma_net
import Constants
from Objects.Map import Map
from Objects.Player import Player
from AI.SingleLayerNet import SingleLayerNet
from AI.reinforce import sample_action_and_stats

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
        # action, logp, entropy = sample_action_and_stats(self.net.forward(input))
        # mu = torch.sigmoid(logits) * 2 - 1
        # sigma = torch.ones_like(mu) * 0.5

        # # Определяем нормальное распределение
        # dist = torch.distributions.Normal(mu, sigma)

        # # Семплируем действие (используем rsample для дифференцируемого пути)
        # action = dist.rsample()

        # # Ограничиваем диапазон действий (например, [-1, 1]) через tanh
        # action = torch.tanh(action)

        # # Вычисляем логарифм вероятности и энтропию
        # logp = dist.log_prob(action).sum(-1)
        # entropy = dist.entropy().sum(-1)

        # a = action.tolist()
        # with torch.no_grad():
        # self.map.ball_teams[self.index][0].velocity[0] = self.map.ball_teams[0][0].velocity[0] + Constants.speed_increment * action[0]
        # self.map.ball_teams[self.index][0].velocity[1] = self.map.ball_teams[0][0].velocity[1] + Constants.speed_increment * action[1]
        # self.map.ball_teams[self.index][0].set_kicking(action[2] > 0)

        return action, logp, entropy
        

        
            
                

