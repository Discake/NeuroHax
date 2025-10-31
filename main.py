import torch
from AI.Maksigma_net import Maksigma_net
from App import App

import Constants


if __name__ == '__main__':
    app = App(play=True, train=False, draw=True, logging=True)

    app.train = False
    app.start_ai_game()


    # mp.set_start_method('spawn', force=True)

    # training = app.training(app.map)
    # training.train(True)