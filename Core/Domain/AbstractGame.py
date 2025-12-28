from abc import ABC, abstractmethod

from Core.Domain.IDrawing import IDrawing
from Core.Domain.Physics import Physics
from Core.Domain.PlayerInput import PlayerInput

class AbstractGame(ABC):
    id : str

    @abstractmethod
    def update(self, physics : Physics):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def handle_input(self, input : PlayerInput):
        pass

    @abstractmethod
    def get_all_players_ids(self) -> tuple[list[int], list[int]]:
        pass

    @abstractmethod
    def load(self):
        pass
    @abstractmethod
    def render(self, drawing : IDrawing):
        pass