from Core.Domain.AbstractGame import AbstractGame
from Core.Domain.IGameRepository import IGameRepository


class ArrayGameRepository(IGameRepository):
    def __init__(self):
        self.games : list[AbstractGame] = []

    def save(self, game):
        if game not in self.games:
            self.games.append(game)
            game.id = len(self.games) - 1

    def get_by_id(self, game_id):
        return self.games[int(game_id)]
