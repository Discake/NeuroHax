from Core.Domain.AbstractGame import AbstractGame
from Core.Domain.Game import Game
from Core.Domain.GameConfig import GameConfig

class AbstractGameFactory():
    @staticmethod
    def create_game(config : GameConfig) -> AbstractGame:
        return Game(id=None, config=config)