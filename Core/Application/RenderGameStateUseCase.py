from Core.Domain.IDrawing import IDrawing
from Core.Domain.IGameRepository import IGameRepository


class RenderGameStateUseCase():
    def __init__(self, repo : IGameRepository):
        self.repo = repo

    def execute(self, game_id : str, drawing : IDrawing):
        game = self.repo.get_by_id(game_id)
        state = game.get_state()
        drawing.draw_state(state)