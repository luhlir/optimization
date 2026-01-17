from bracketMethod import BracketMethod


class QuadraticFitSearch(BracketMethod):

    def search(self, f, lower, upper, **kwargs):
        """
        Interpolates a quadratic function around a, b, and c and brackets that min.

        a < b (< c if it is not None)

        :param f: Function that takes a constant input and returns a constant output
        :param lower: Point around local min
        :param middle: Point around local min (> a)
        :param upper: Point around local min (> b), defaults to midpoint of a and b
        :return: Tuple with 2 values that bracket a local min of f
        """
        steps = self.steps
        if kwargs["middle"] is None:
            middle = (lower + upper) / 2
        else:
            middle = kwargs["middle"]
        y_lower, y_middle, y_upper = f(lower), f(middle), f(upper)
        steps -= 3
        while steps > 0:
            x = y_lower * (middle ** 2 - upper ** 2) + y_middle * (upper ** 2 - lower ** 2) + y_upper * (lower ** 2 - middle ** 2)
            x /= y_lower * (middle - upper) + y_middle * (upper - lower) + y_upper * (lower - middle)
            x /= 2
            yx = f(x)
            if x > middle:
                if yx > y_middle:
                    upper, y_upper = x, yx
                else:
                    lower, y_lower, middle, y_middle = middle, y_middle, x, yx
            elif x < middle:
                if yx > y_middle:
                    lower, y_lower = x, yx
                else:
                    upper, y_upper, middle, y_middle = middle, y_middle, x, yx
            else:
                break
            steps -= 1
        return lower, upper
