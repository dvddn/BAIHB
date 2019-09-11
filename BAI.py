import numpy as np
from Hyperband import HyperBand
from Worker import Worker
#from KDE import DensityEstimator
import numpy as np
from Arm import Arm

class BAI(object):
    def __init__(self, n, params, R, eta, date):
        self.n = n
        self.arms = []*n
        self.best = 0

        space = np.logspace(np.log10(params['eta'][0]), np.log10(params['eta'][1]), self.n)

        for i in range(n):
            params2 = params
            params2[eta] = [space[i],space[i+1]]
            model = Worker(date, params2)
            self.arms[i] = Arm(HyperBand(model, params, R, eta),)
        return

    def get_next_arm(self):
        return self.probabilities.index(max(self.probabilities()))

    def run_arm(self, i):
        self.arms[i].run()
        # testing commit

