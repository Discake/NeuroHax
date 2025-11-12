from Core.Objects.Map import Map
from Player_actions.Player_action import Player_action
from AI.Translator import Translator

class Net_action(Player_action):
    def __init__(self, map : Map, net, player, is_team_1):
        self.player = player

        self.translator = Translator(map, net, self.player, is_team_1)
        self.net = net
        self.is_team1 = is_team_1
    
    def act(self, input_to_act):
        
        super().act(input_to_act)
