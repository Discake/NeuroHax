if __name__ == '__main__':

    import torch.multiprocessing as mp
    from AI.Maksigma_net import Maksigma_net
    from App import App

    do_training = False

    def training():
        app = App(play=False, draw_game=False, logging=True)
        app.training(max_steps=1024, draw_stats=True, load_filename="Maksigma_net_ravnykh_new_method_3.pth", save_filename="test.pth")

    def ai():
        app = App(play=True, draw_game=True, logging=True)
        app.start_ai_game("test.pth")


    mp.set_start_method('spawn')

    if do_training:
        training()
    else:
        ai()



