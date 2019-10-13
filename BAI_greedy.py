import numpy as np
from Hyperband import HyperBand
from Worker_NEW import Worker
#from KDE import DensityEstimator
import numpy as np
from Arm_greedy import Arm
import pickle
from scipy.stats import uniform, randint

class BAI(object):
    def __init__(self, n, params, interval, R, eta, eps):
        self.n = n
        self.arms = [None]*n
        self.best = 0
        self.gen_1 = uniform(0,1)
        self.gen_2 = randint(0,n-1)
        self.eps = eps
        space = np.logspace(np.log10(interval[0]), np.log10(interval[1]), self.n+1)
        data = self.get_data()
        
        for i in range(n):
            params2 = params.copy()
            params2.update({'eta':uniform(space[i],space[i+1]-space[i])})
            model = Worker(params2, data)
            print('Arm ', i, 'will contain Hyperband object with eta in [', space[i],',',space[i+1],']')
            self.arms[i] = Arm(HyperBand(model, params, R, eta))
        return

    def get_next_arm(self):
        maxproba = [0,-1]
        for i in range(self.n):
            if self.arms[i].best_mean > maxproba[0]:
                maxproba = [self.arms[i].best_mean, i]
        if self.gen_1.rvs() < 1-self.eps:
            return maxproba[1]
        else:
            tmp = set(range(self.n))
            tmp.remove(maxproba[1])
            return list(tmp)[self.gen_2.rvs()]


    def run_arm(self, i):
        print("RUNNING ARM ", i)
        self.arms[i].hb.run()
        self.arms[i].compute_best_mean()
        print(self.arms[i].hb.evals)
        return
    
    def run_n(self, n):
        for i in range(n):
            j = self.get_next_arm()
            self.run_arm(j)
        return
    
    def get_data(self):
        return pickle.load(open("mydata.p", "rb"))