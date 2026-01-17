from lineSearch import LineSearch
from architecture_temp.bracketMethods.bracketEnum import BracketEnum
from collections.abc import Callable


class FullSearch(LineSearch):

    def __init__(self, objective_function: Callable, bracket_method: BracketEnum = BracketEnum.GOLDEN_SECTION_SEARCH,
                 bracket_args: dict = {}, bracket_min_args: dict = {}):
        super().__init__(objective_function)
        self.bracketSearch = bracket_method.value(bracket_args["steps"])
        self.bracketSearchArgs = bracket_args
        if bracket_method is not BracketEnum.BRACKET_MINIMUM:
            self.bracketMin = BracketEnum.BRACKET_MINIMUM.value(bracket_min_args["steps"])
            self.bracketMinArgs = bracket_min_args
        else:
            self.bracketMin = self.bracketSearch
            self.bracketMinArgs = self.bracketSearchArgs

    def __search(self, x, d):
        def f_wrap(a):
            return self.objective_function(x + a * d)

        b, c = self.bracketMin.search(f_wrap, **self.bracketMinArgs)
        if self.bracketMin is self.bracketSearch:
            y, z = b, c
        else:
            y, z = self.bracketSearch.search(f_wrap, b, c, **self.bracketSearchArgs)
        return (y + z) / 2
