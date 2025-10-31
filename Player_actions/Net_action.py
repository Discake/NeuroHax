import torch
import Constants
from Objects.Map import Map
from Player_actions.Player_action import Player_action
from AI.Translator import Translator
from AI.Maksigma_net import Maksigma_net

class Net_action(Player_action):
    def __init__(self, nn : Maksigma_net, map : Map):
        self.nn = nn
        self.translator = Translator(map, nn)
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
        self.player.set_kicking(input[2] > 0)