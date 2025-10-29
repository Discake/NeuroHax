from App import App

if __name__ == '__main__':
    app = App(play=False, training=True, draw=True, logging=True)
    app.start_ai_game()