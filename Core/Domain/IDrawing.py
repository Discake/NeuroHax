from abc import ABC, abstractmethod

class IDrawing(ABC):
    @abstractmethod
    def draw_state(self, state : dict):
        pass