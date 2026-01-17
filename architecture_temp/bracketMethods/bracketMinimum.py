from bracketMethod import BracketMethod


class BracketMinimum(BracketMethod):

    def search(self, f, **kwargs):
        """
        Guarantees a local min of f is between the returned values. Does not guarantee a small bracket.

        :param f: Function takes a constant input and returns a constant output
        :param step: Initial step size
        :param growth: Growth factor by which the step increases
        :return: Tuple with 2 values that bracket a local min of f
        """
        step = kwargs["step_size"]
        growth = kwargs["growth"]
        max_steps = self.steps
        a = 0
        b = self.steps
        c = 0
        ya, yb = f(a), f(b)
        max_steps -= 2
        if ya < yb:
            temp = a
            a = b
            b = temp
            yb = ya
            step = -step
        while max_steps > 0:
            c = b + step
            yc = f(c)
            if yc > yb:
                break
            a = b
            b = c
            yb = yc
            step *= growth
            max_steps -= 1
        return a, c
