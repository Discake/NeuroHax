from abc import ABC, abstractmethod

class TranslatorFromPolicy(ABC):
    @abstractmethod
    def get_action(self, policy, state : dict) -> list:
        pass