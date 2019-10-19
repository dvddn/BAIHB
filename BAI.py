import numpy as np
from Hyperband import HyperBand
from Worker_NEW import Worker
#from KDE import DensityEstimator
import numpy as np
from Arm import Arm
import pickle
from scipy.stats import uniform

class BAI(object):
    def __init__(self, n, params, interval, R, eta):
        self.n = n
        self.arms = [None]*n
        self.best = 0

        space = np.logspace(np.log10(interval[0]), np.log10(interval[1]), self.n+1)
        data = self.get_data()
        
        for i in range(n):
            params2 = params.copy()
            params2.update({'eta':uniform(space[i],space[i+1]-space[i])})
            model = Worker(params2, data)
            print('Arm ', i, 'will contain Hyperband object with eta in [', space[i],',',space[i+1],']')
            self.arms[i] = Arm(HyperBand(model, params, R, eta), 0.5, 1, 1, 1)
        return

    def get_next_arm(self):
        maxproba = [-1,-1]
        for i in range(self.n):
            if self.arms[i].improvement > maxproba[0]:
                maxproba = [self.arms[i].improvement, i]
        return maxproba[1]

    def run_arm(self, i):
        print("RUNNING ARM ", i)
        self.arms[i].hb.run()
        self.arms[i].compute_posterior()
        
        if (max(self.arms[i].hb.evals['L']) > self.best):
            self.best = max(self.arms[i].hb.evals['L'])
            print("NEW BEST: ", self.best)
        
        [x.compute_probability(self.best) for x in self.arms]
        print(self.arms[i].hb.evals)
        return
    
    def run_n(self, n):
        for i in range(n):
            j = self.get_next_arm()
            self.run_arm(j)
        return
    
    def get_data(self):
        return pickle.load(open("mydata.p", "rb"))
