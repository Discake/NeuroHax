
from Core.Domain.Physics import Physics
from Core.Domain.IGameRepository import IGameRepository
from Core.Domain.PlayerInput import PlayerInput
from Core.Infrastructure.PlayerController import PlayerController

class UpdateGameStateUseCase():
    def __init__(self, game_repository : IGameRepository, player_controllers : list[PlayerController]):
        self.game_repository = game_repository
        self.player_controllers = player_controllers

    def execute(self, game_id, physics : Physics):
        game = self.game_repository.get_by_id(game_id)
        for player in self.player_controllers:
            if not player.is_acting():
                continue
            input = player.get_action()
            game.handle_input(input)
        game.update(physics)
        self.game_repository.save(game)

    