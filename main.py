from App import App
import multiprocessing as mp


if __name__ == '__main__':
    app = App(play=False, train=True, draw=True, logging=True)
    # app.start_ai_game()
    mp.set_start_method('spawn', force=True)

    training = app.training(app.map)
    training.train(True)