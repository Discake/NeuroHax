import torch
import Constants
from Objects.Map import Map
from Player_actions.Player_action import Player_action
from AI.Translator import Translator
from AI.Maksigma_net import Maksigma_net

class Net_action(Player_action):
    def __init__(self, nn : Maksigma_net, map : Map, index):
        self.nn = nn
        self.translator = Translator(map, nn, index)
        self.map = map
        self.hit_flag = False
        self.count = 1

    def update_translator(self):
        self.hit_flag = False
        input = self.translator.translate_input()
        self.translator.translate_output(input) 
        
    def act(self, input):
        input = torch.squeeze(input, 0)
        super().act(input)
        # self.update_translator()
        
        # output = self.translator.output
        # self.player.velocity.x += Constants.speed_increment * self.translator.output[0]
        # self.player.velocity.y += Constants.speed_increment * self.translator.output[1]
        # self.player.set_kicking(self.translator.output[2] > 0)

        # shape = output.shape

        # shape = shape[:-1]

        # shape = torch.tensor(shape)
        # for i in shape:
        #      i -= 1
        # shape = torch.Size(shape)

        # if input.shape[0] == 1:
        #     input = input.squeeze(0)
        
        # list1 = [self.player.velocity.x, self.player.velocity.y]
        # list2 = [input[0], input[1]]

        # for i in range(2):
        #     if isinstance(list2[i], torch.Tensor):
        #         # if abs(list2[i]) > 0.6:
        #             list1[i] = list2[i] * Constants.speed_increment
        #     else:
        #         # if abs(list2[i]) > 0.6:
        #             list1[i] = list2[i] * Constants.speed_increment

        # if(isinstance(list[0], torch.Tensor)):
        #     self.player.velocity.x = list1[0]
        #     self.player.velocity.y = list1[1]
        # else:
        #     self.player.velocity.x = list1[0]
        #     self.player.velocity.y = list1[1]
        

        # self.player.velocity.x += Constants.speed_increment * output.tolist()[0]
        # self.player.velocity.y += Constants.speed_increment * output.tolist()[1]
        self.player.set_kicking(input[2] > 0)


        # self.x_vel_sum += self.player.velocity.x
        # self.y_vel_sum += self.player.velocity.y
        # print(f"mean player speed x {self.x_vel_sum / self.count}")
        # print(f"mean player speed y {self.y_vel_sum / self.count}")
        # self.count += 1