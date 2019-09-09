import numpy as np
from Hyperband import HyperBand
from Worker import Worker
from KDE import DensityEstimator

class BAI(object):
    def __init__(self, n, params, R, eta, date):
        self.n = n
        self.arms = []*n
        self.kdes = []*n
        self.probabilities = [0]*n
        self.best = 0

        for i in range(n):
            model = Worker(date, params)
            params2 = params
            self.arms[i] = HyperBand(model, params, R, eta)
            self.kdes[i] = DensityEstimator( ... )
            ####TODO

        return

