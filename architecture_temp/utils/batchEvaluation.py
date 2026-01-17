import multiprocessing


class BatchEval:

    def __init__(self, multithreaded):
        if multithreaded:
            self.pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
        else:
            self.pool = None

    def __del__(self):
        if self.pool is not None:
            self.pool.close()

    @staticmethod
    def __multiprocessing_eval(f, x):
        return f(x)

    def eval(self, f, X):
        if self.pool is not None:
            args = []
            for x in X:
                args.append((f, x))
            y = self.pool.starmap(BatchEval.__multiprocessing_eval, args)
        else:
            y = []
            for i in range(len(X)):
                y.append(f(X[i]))
        return y
