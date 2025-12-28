from abc import ABC, abstractmethod
from Core.Domain.AbstractGame import AbstractGame

class IGameRepository(ABC):
    @abstractmethod
    def get_by_id(self, game_id: str) -> AbstractGame:  # Возвращает агрегат целиком
        pass
    
    @abstractmethod
    def save(self, game: AbstractGame) -> None:  # Сохраняет агрегат целиком
        pass
