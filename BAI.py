import numpy as np
from Hyperband import HyperBand
from Worker_NEW import Worker
#from KDE import DensityEstimator
import numpy as np
from Arm import Arm
import pickle

class BAI(object):
    def __init__(self, n, params, R, eta):
        self.n = n
        self.arms = []*n
        self.best = 0

        space = np.logspace(np.log10(params['eta'][0]), np.log10(params['eta'][1]), self.n)
        data = self.get_data()
        
        for i in range(n):
            params2 = params
            params2[eta] = [space[i],space[i+1]]
            model = Worker(params2, data)
            self.arms[i] = Arm(HyperBand(model, params, R, eta), 0.5, 1, 1, 1)
        return

    def get_next_arm(self):
        return self.probabilities.index(max(self.probabilities()))

    def run_arm(self, i):
        self.arms[i].hb.run()
        self.arms[i].compute_posterior()
        print("RUNNING ARM ", i)
        if (max(self.arms[i].hb.evals['L']) > self.best):
            self.best = max(self.arms[i].hb.evals['L'])
            [x.compute_probability(self.best) for x in self.arms]

        else:
            self.arms[i].compute_probability(self.best)
        
        return
    
    def run_n(self, n):
        for i in range(n):
            j = self.get_next_arm()
            self.run_arm(j)
        return
    
    def get_data(self):
        return pickle.load(open("mydata.p", "rb"))