from abc import ABC, abstractmethod

from Core.Domain.PlayerInput import PlayerInput

class PlayerController(ABC):
    def __init__(self, player_id, team_id):
        self.player_id = player_id
        self.team_id = team_id

    @abstractmethod
    def is_acting(self):
        pass

    @abstractmethod
    def get_action(self) -> PlayerInput:
        pass