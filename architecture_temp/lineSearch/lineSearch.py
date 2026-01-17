from abc import ABC, abstractmethod
from collections.abc import Callable


class LineSearch(ABC):

    def __init__(self, objective_function: Callable):
        self.objective_function = objective_function

    def search(self, x, d):
        """
        Calls the specific line search method
        """
        if len(x) != len(d):
            return x

        a = self.__search(x, d)
        return x + a * d

    @abstractmethod
    def __search(self, x, d):
        pass
