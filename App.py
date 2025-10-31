import pygame
from AI.Maksigma_net import Maksigma_net
from Draw.Drawing import Drawing
from Objects.Map import Map
from Objects.Gate import Gate
from Data_structure.Gates_data import Gates_data
from Physics.WallCollision import WallCollision as Wall
import Constants
from Player_actions.Keydown_action import Keydown_action
from Player_actions.Net_action import Net_action
from AI.SingleLayerNet import SingleLayerNet
import copy
from threading import Timer
from AI.Environment import Environment
import torch
from AI.training import Training

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.set_num_threads(14)   # например, 8 потоков/ядер
torch.set_num_interop_threads(14)
has = torch.has_mkl

device = Constants.device

class App:
    def __init__(self, play, train, draw = False, logging = True):
        # Initialize Pygame
        self.draw = draw
        self.logging = logging
        if draw:
            pygame.init()

        self.play = play
        self.train = train

        self.map = Map()
        self.drawing = Drawing(self.map)

        pygame.display.set_caption("NeuroHax")

    def start_single_game(self):
        
        running = True
        
        keydown_action = Keydown_action()
        keydown_action2 = Keydown_action()
        actions = [keydown_action, keydown_action2]

        # Game loop
        while running:
            
            pygame.time.delay(5)  # Pause for 5 milliseconds

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                for action in actions:
                    action.set_event(event)
            
            for i in range(len(actions)):
                actions[i].set_player(self.map.players[i])
                actions[i].act()

            self.drawing.draw()
            self.map.move_balls()  # Move balls and resolve collisions
            pygame.display.update()

        # Quit Pygame
        pygame.quit()

    def start_ai_game(self, save_filename = None):
        running = True

        actions = []
        
        if not self.train:
            for i in range(Constants.player_number - 1 if self.play else Constants.player_number):
                nn = Maksigma_net()
                if(save_filename is not None):
                    nn.load_state_dict(torch.load(save_filename))
                nn = nn.eval()
                ai_action = Net_action(nn, self.map)
                ai_action.set_player(self.map.players_team1[i])
                actions.append(ai_action)
            
        if self.play:
            keydown_action = Keydown_action()
            actions.append(keydown_action)
            keydown_action.set_player(self.map.players_team1[1])

        if self.train:
            # torch.autograd.set_detect_anomaly(True)
            nn, save = self.training(self.map, draw=self.draw)

            torch.save({'model_state_dict': save}, f'{nn.name}.pth')

            ai_action = Net_action(nn, self.map, 0)
            actions.append(ai_action)  

        if self.draw:
            # Game loop
            while running:
                pygame.time.delay(7)  # Pause for 5 milliseconds

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    for action in actions:
                        if isinstance(action, Keydown_action):
                            action.set_event(event)

                for action in actions:
                    if isinstance(action, Net_action):
                        inp, _, _ = action.translator.translate_output(action.translator.translate_input())
                        action.act(inp)
                    else:
                        action.act()

                    


                self.drawing.draw()
                self.map.move_balls()  # Move balls and resolve collisions               

                pygame.display.update()

            # Quit Pygame
            pygame.quit()

    def callback(self):
        self.drawing.draw()
        pygame.display.update()
        t = Timer(0.5, self.callback)
        t.start()

    def training(self, save_filename = None, draw_stats = False):

        model = Maksigma_net().to(device=device)

        if save_filename is not None:
            checkpoint = torch.load(save_filename)
            model.load_state_dict(checkpoint)
            print(f"Model loaded successfully from: {save_filename}")
        
        print(f"Computation on device: {device}")
        

        t = Training(Environment(self.map, model), draw_stats=draw_stats)
        
        return t.train(draw_stats)
