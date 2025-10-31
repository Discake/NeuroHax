import torch
from AI.Maksigma_net import Maksigma_net
from App import App

import Constants


if __name__ == '__main__':
    app = App(play=False, train=True, draw=True, logging=True)

    app.train = False
    app.start_ai_game()


    # mp.set_start_method('spawn', force=True)

    # training = app.training(app.map)
    # training.train(True)

    checkpoint = torch.load(f'Maksigma_net_ravnykh_new_method.pth')
    model = Maksigma_net().to(device=Constants.device)
    print(f"Computation on device: {Constants.device}")  # Создайте экземпляр модели
    model.load_state_dict(checkpoint)

    result = app.training(model, draw_stats=True)
    torch.save(model.state_dict(), f'Maksigma_net_ravnykh_new_method.pth')