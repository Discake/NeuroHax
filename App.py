from AI.Maksigma_net import Maksigma_net
from AI.SeparateNetworkPolicy import SeparateNetworkPolicy
from AI.Simple import UltraSimplePolicy
from Core.Objects.Map import Map
import Constants
from Player_actions.Keydown_action import Keydown_action
from Player_actions.Net_action import Net_action
from Draw.Drawing import Drawing
import pygame
from threading import Timer
from AI.Training.Environment import Environment
import torch
from AI.Training.Training_process import Training_process

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.set_num_threads(14)   # например, 8 потоков/ядер
torch.set_num_interop_threads(14)
has = torch.has_mkl

device = Constants.device

class App:
    def __init__(self, play, draw_game = False, drawStats = False, logging = True):
        
        self.draw_game = draw_game
        self.logging = logging
        self.map = Map()        
        if draw_game:
            pygame.init()
            pygame.display.set_caption("NeuroHax")
            self.drawing = Drawing(self.map)

        self.play = play


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

    def start_ai_game(self, load_filename = None):
        running = True

        actions = list[Keydown_action|Net_action]()
        
        for i in range(Constants.player_number):
            nn = SeparateNetworkPolicy()
            if(load_filename is not None):
                nn.load_state_dict(torch.load(load_filename))
            nn = nn.eval()
            ai_action = Net_action(self.map, nn, self.map.players_team1[i], is_team_1=True)
            actions.append(ai_action)
            
        for i in range(0 if self.play else Constants.player_number):
            nn = SeparateNetworkPolicy()
            if(load_filename is not None):
                nn.load_state_dict(torch.load(load_filename))
            nn = nn.eval()
            ai_action = Net_action(self.map, nn, self.map.players_team2[i], is_team_1=False)
            actions.append(ai_action)
            
        if self.play:
            keydown_action = Keydown_action()
            actions.append(keydown_action)
            keydown_action.set_player(self.map.players_team2[0])  

        if self.draw_game:
            # Game loop
            while running:
                pygame.time.delay(2)  # Pause for 5 milliseconds

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    for action in actions:
                        if isinstance(action, Keydown_action):
                            action.set_event(event)

                for action in actions:
                    if isinstance(action, Net_action):
                        input_to_net = action.translator.translate_input()
                        input_to_act, _  = action.net.select_action(input_to_net)

                        # if not action.is_team1:
                        #     input_to_act[0] *= -1
                        action.act(input_to_act)
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

    def training(self, max_steps, load_filename = None, save_filename = None, draw_stats = False):

        model1 = SeparateNetworkPolicy().to(device=device)
        model2 = SeparateNetworkPolicy().to(device=device)

        if load_filename is not None:
            checkpoint = torch.load(load_filename)
            model1.load_state_dict(checkpoint)
            print(f"Model loaded successfully from: {load_filename}")
            checkpoint = torch.load(load_filename)
            model2.load_state_dict(checkpoint)
            print(f"Model loaded successfully from: {load_filename}")
        
        print(f"Computation on device: {device}")
        

        t = Training_process(Environment(model1, model2, num_steps=max_steps), draw_stats=draw_stats)
        
        return t.train(max_steps_per_worker=max_steps, save_filename=save_filename)
