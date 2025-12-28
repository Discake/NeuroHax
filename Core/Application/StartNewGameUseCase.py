
from Core.Domain.AbstractGameFactory import AbstractGameFactory
from Core.Domain.GameConfig import GameConfig
from Core.Domain.IGameRepository import IGameRepository

class StartNewGameUseCase():
    def __init__(self, game_repository : IGameRepository):
        self.game_repository = game_repository

    def execute(self, config : GameConfig):
        '''
        :return: id of created game
        '''
        game = AbstractGameFactory.create_game(config)
        game.load()
        self.game_repository.save(game)

        return game.id, game.get_all_players_ids()
