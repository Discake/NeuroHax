from Core.Objects.Map import Map
from Player_actions.Player_action import Player_action
from AI.Translator import Translator
from AI.Maksigma_net import Maksigma_net

class Net_action(Player_action):
    def __init__(self, map : Map, net : Maksigma_net, player):
        self.player = player

        self.other_players = []
        for other_player in map.players_team1 + map.players_team2:
            if other_player != self.player:
                self.other_players.append(other_player)


        self.translator = Translator(map, net, self.player, self.other_players)
        self.net = net
    
    def act(self):
        input_to_net = self.translator.translate_input()
        input_to_act, _, _, _ = self.net.get_action(input_to_net)
        super().act(input_to_act)
