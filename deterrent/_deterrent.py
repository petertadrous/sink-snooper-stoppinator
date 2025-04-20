from abc import ABC, abstractmethod


class Deterrent(ABC):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def activate(self, duration: float):
        pass

    @abstractmethod
    def cleanup(self):
        pass
