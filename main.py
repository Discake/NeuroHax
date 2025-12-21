if __name__ == '__main__':

    import torch.multiprocessing as mp
    from App import App

    do_training = False

    def training():
        app = App(play=False, draw_game=False, logging=True)
        app.training(max_steps=4000, draw_stats=True, \
                     load_filename=None, \
                      save_filename="Checkpoints/separate_goals_v9.pth")

    def ai():
        app = App(play=False, draw_game=True, logging=True)
        app.start_ai_game("Checkpoints/separate_goals_v9.pth")


    mp.set_start_method('spawn')
    


    if do_training:
        training()
    else:
        ai()    
