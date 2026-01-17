from abc import ABC, abstractmethod


class BracketMethod(ABC):

    def __init__(self, steps=10):
        self.steps = steps

    @abstractmethod
    def search(self, f, lower, upper, **kwargs):
        pass
