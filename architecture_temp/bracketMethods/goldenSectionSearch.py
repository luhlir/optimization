from bracketMethod import BracketMethod


class GoldenSectionSearch(BracketMethod):

    def search(self, f, lower, upper, **kwargs):
        """
        Uses the golden ration to shrink the bracket at each step

        :param f: Function that takes a constant input and returns a constant output
        :param lower: Constant lower bound of bracket
        :param upper: Constant upper bound of bracket
        :param steps: Number of function calls (>1)
        :return: Tuple with 2 values that bracket a local min of f
        """
        steps = self.steps
        p = 0.61803  # golden ration - 1
        d = p * upper + (1 - p) * lower
        yd = f(d)
        steps -= 1
        while steps > 0:
            c = p * lower + (1 - p) * upper
            yc = f(c)
            if yc < yd:
                upper = d
                d = c
                yd = yc
            else:
                lower = upper
                upper = c
            steps -= 1
        return lower, upper
