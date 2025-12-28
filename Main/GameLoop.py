from Core.Application.RenderGameStateUseCase import RenderGameStateUseCase
from Core.Application.StartNewGameUseCase import StartNewGameUseCase
from Core.Application.UpdateGameStateUseCase import UpdateGameStateUseCase
from Core.Domain.GameConfig import GameConfig
from Core.Domain.IDrawing import IDrawing
from Core.Domain.IGameRepository import IGameRepository
from Core.Domain.Physics import Physics
from Core.Infrastructure.PlayerController import PlayerController


class GameLoop():
    def __init__(self, repo : IGameRepository, physics : Physics, game_config : GameConfig, drawing : IDrawing = None):
        self.repo = repo
        start_use_case = StartNewGameUseCase(repo)
        self.game_id, players_ids = start_use_case.execute(game_config)
        self.players_red_ids = players_ids[0]
        self.players_blue_ids = players_ids[1]
        self.physics = physics
        self.drawing = drawing
        self.player_controllers : list[PlayerController] = []

    def set_player_controller(self, controller : PlayerController):
        self.player_controllers.append(controller)


    def run(self):
        while True:
            update_use_case = UpdateGameStateUseCase(self.repo, self.player_controllers)
            update_use_case.execute(self.game_id, self.physics)
            if self.drawing is not None:
                render_use_case = RenderGameStateUseCase(self.repo)
                render_use_case.execute(self.game_id, self.drawing)
