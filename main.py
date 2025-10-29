from App import App

if __name__ == '__main__':
    app = App(play=False, train=True, draw=False, logging=True)
    # app.start_ai_game()
    training = app.training(app.map)
    training.train(True)