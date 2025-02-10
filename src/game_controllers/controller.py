from abc import ABC, abstractmethod


class Controller(ABC):
    @abstractmethod
    def get_response(self, game) -> tuple[int, int, bool]:
        pass