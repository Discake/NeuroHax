import torch.multiprocessing as mp
from AI.Maksigma_net import Maksigma_net
from App import App

import Constants

train = False

def training(app : App):
    Constants.time_increment = 1
    app.train = True
    _ = app.training(save_filename="Maksigma_net_ravnykh_new_method_2.pth", draw_stats=True)

def ai(app : App):
    Constants.time_increment = 1
    app.play = True
    app.start_ai_game("Maksigma_net_ravnykh_new_method_2.pth")

if __name__ == '__main__':
    app = App(play=False, train=False, draw=True, logging=True)

    if train:
        training(app)
    else:
        ai(app)



