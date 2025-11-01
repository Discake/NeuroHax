from Player_actions.Player_action import Player_action
from AI.Translator import Translator

class Net_action(Player_action):
    def __init__(self, map, net):
        self.translator = Translator(map, net)
    
    def act(self):
        input = self.translator.translate_input()
        super().act(input)
