if __name__ == '__main__':

    import torch.multiprocessing as mp
    from App import App

    do_training = False

    def training():
        app = App(play=False, draw_game=False, logging=True)
        app.training(max_steps=2048, draw_stats=True, \
                     load_filename=None,\
                      save_filename="Checkpoints/test_1_v_1.pth")

    def ai():
        app = App(play=True, draw_game=True, logging=True)
        app.start_ai_game("Checkpoints/test_1_v_1.pth")


    mp.set_start_method('spawn')

    if do_training:
        training()
    else:
        ai()    
