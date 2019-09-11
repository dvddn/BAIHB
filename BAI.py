import numpy as np
from Hyperband import HyperBand
from Worker import Worker
#from KDE import DensityEstimator
import numpy as np

class BAI(object):
    def __init__(self, n, params, R, eta, date):
        self.n = n
        self.arms = []*n
        #self.kdes = []*n
        self.means = [0.5]*n
        self.stds = [1]*n
        self.probabilities = [1000]*n
        self.best = 0

        space = np.logspace(np.log10(params['eta'][0]), np.log10(params['eta'][1]), self.n)

        for i in range(n):
            model = Worker(date, params)
            params2 = params
            params2[eta] = [space[i],space[i+1]]
            self.arms[i] = HyperBand(model, params, R, eta)
        return

    def get_next_arm(self):
        return self.probabilities.index(max(self.probabilities()))

    def run_arm(self, i):
        self.arms[i].run()
        # testing commit

