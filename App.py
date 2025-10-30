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
from AI.reinforce import reinforce
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
        self.train_loader = None
        self.draw = draw
        self.logging = logging
        if draw:
            pygame.init()

        self.play = play
        self.train = train

        self.map = Map(Constants.field_size[0], Constants.field_size[1])
        self.drawing = Drawing(self.map)

        self.map.add_players()
        self.map.add_balls()
        
        teams = []
        
        self.teams_number = 10

        if train or play:
            self.teams_number = 1

        for i in range(self.teams_number):
            teams.append(copy.deepcopy(self.map.balls))

        self.map.set_ball_teams(teams)

        self.add_gates()
        self.add_walls()

        pygame.display.set_caption("Hello MAZAFAKA")
        self.map.save_state()

        self.traj_list = []
        
        for _ in range(self.teams_number):
            self.traj_list.append({"logps" : float, "rewards" : []})

    def add_gates(self):
        gates_data = [Gates_data(is_left=True), Gates_data(is_left=False)]
        for data in gates_data:
            gate = Gate(data)
            # gate.set_screen(self.drawing.screen)
            gate.add_boundaries()
            self.map.add_gate(gate)

    def add_walls(self):
        walls = [Constants.wall1, Constants.wall2, Constants.wall3, Constants.wall4, Constants.wall5, 
                 Constants.wall6, Constants.wall7, Constants.wall8, Constants.wall9, Constants.wall10]
        for wall in walls:
            self.map.add_wall(Wall(wall.start, wall.end, wall.constant, wall.is_vertical))

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

    def start_ai_game(self):
        running = True

        actions = []
        
        if not self.play and not self.train:
            for j in range(self.teams_number):
                for i in range(Constants.player_number - 1 if self.play and j == 0 else Constants.player_number):
                    # if i == 0 and j == 0:
                    #     nn = self.train()
                    nn = Maksigma_net()
                    nn.load_state_dict(torch.load(f'{nn.name}.pth'))
                    nn = nn.eval()
                    ai_action = Net_action(nn, self.map, j)
                    actions.append(ai_action)
            
        if self.play:
            keydown_action = Keydown_action()
            actions.append(keydown_action)

        if self.train:
            # torch.autograd.set_detect_anomaly(True)
            nn, save = self.training(self.map, draw=self.draw)

            torch.save({'model_state_dict': save}, f'{nn.name}.pth')

            ai_action = Net_action(nn, self.map, 0)
            actions.append(ai_action)  

        if self.draw:
            # Game loop
            while running:
                pygame.time.delay(5)  # Pause for 5 milliseconds

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    for action in actions:
                        if isinstance(action, Keydown_action):
                            action.set_event(event)
                
                # for i in range(len(actions)):
                #     if isinstance(actions[i], Net_action):
                #         actions[i].update_translator()

                for j in range(self.teams_number):
                    for i in range(Constants.player_number):
                        action = actions[i + j * Constants.player_number]

                        if isinstance(action, Net_action):
                            action.set_player(self.map.ball_teams[j][i])
                            # action.update_translator()
                            inp, _, _ = action.translator.translate_output(action.translator.translate_input())
                            action.act(inp)
                        else:
                            action.set_player(self.map.ball_teams[0][0])
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

    def training(self, map, draw = False):
        checkpoint = torch.load('Maksigma_net_ravnykh_2.pth')
        model = Maksigma_net().to(device=device)
        print(f"Device = {device}")  # Создайте экземпляр модели
        model.load_state_dict(checkpoint)
        # # loss = checkpoint['loss']
        # # r = reinforce(model, lr=3e-2, loss=10)
        # # r.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # # maksigma = Maksigma_net()
        # r = reinforce(model, lr=3e-4, loss=10)

        # nn = r.train(model, Enviroment(model), episodes=6000000000, batch_episodes=1, gamma=0.995, device="cpu")

        t = Training(Environment(map, model), self.train_loader, draw=draw)
        
        return t
