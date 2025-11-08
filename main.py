if __name__ == '__main__':

    import torch.multiprocessing as mp
    from App import App

    do_training = True

    def training():
        app = App(play=False, draw_game=False, logging=True)
        app.training(max_steps=3072, draw_stats=True, \
                     load_filename="Checkpoints/test_1_v_1_v8.pth",\
                      save_filename="Checkpoints/test_1_v_1_v9.pth")

    def ai():
        app = App(play=False, draw_game=True, logging=True)
        app.start_ai_game("Checkpoints/test_1_v_1_v9.pth")


    mp.set_start_method('spawn')

    if do_training:
        training()
    else:
        ai()    
