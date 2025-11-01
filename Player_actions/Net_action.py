from Player_actions.Player_action import Player_action
from AI.Translator import Translator
from AI.Maksigma_net import Maksigma_net

class Net_action(Player_action):
    def __init__(self, map, net : Maksigma_net):
        self.translator = Translator(map, net)
        self.net = net
    
    def act(self):
        input_to_net = self.translator.translate_input()
        input_to_act, _, _, _ = self.net.get_action(input_to_net)
        super().act(input_to_act)
