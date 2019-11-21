import numpy as np
from Hyperband import HyperBand
from Worker_NEW import Worker
from ConfigGenerator import config_generator
#from KDE import DensityEstimator
import numpy as np
from Arm import Arm
import pickle
from scipy.stats import uniform,randint
import os

class BAI(object):
    def __init__(self, nn, params, interval, R, eta):
        self.n = nn
        self.arms = [None]*self.n
        self.best = 0
        
        #depths = [[2,4],[4,8],[8,15]]
        space = np.logspace(np.log10(interval[0]), np.log10(interval[1]), nn+1)
        data = self.get_data()
        
        for i in range(int(self.n)):
            
                params2 = params.copy()
                #params2.update({'max_depth':randint(depths[j][0],depths[j][1])})
                params2.update({'eta':uniform(space[i],space[i+1]-space[i])})
                model = Worker(params2, data)
                cg = config_generator(params2)
                #print('Arm {} will contain Hyperband object with eta in [{},{}] and depth in [{},{}]'.format(3*i+j,space[i],space[i+1],depths[j][0],depths[j][1]), file=open('logTS.txt','a'))
                self.arms[i] = Arm(HyperBand(model, cg, R, eta), 0.8, 1, 1, 0.005)
        return

    def get_next_arm(self):
        maxproba = [-1,-1]
        for i in range(self.n):
            if self.arms[i].improvement > maxproba[0]:
                maxproba = [self.arms[i].improvement, i]
        return maxproba[1]

    def run_arm(self, i):
        print("RUNNING ARM {}".format(i), file=open('logTS.txt','a'))
        self.arms[i].hb.run()
        self.arms[i].compute_posterior()
        
        if (max(self.arms[i].hb.evals['L']) > self.best):
            self.best = max(self.arms[i].hb.evals['L'])
            print("NEW BEST: {} ".format(self.best), file=open('logTS.txt','a'))
        
        [x.compute_probability(self.best) for x in self.arms]
        print(self.arms[i].hb.evals)
        with open('./RESULTS/Intermediate_Arm_{}'.format(i), 'wb') as handle:
            pickle.dump(self.arms[i].hb.evals, handle)

        return
    
    def run_n(self, n):
        for i in range(n):
            os.popen("echo 'Running iteration {}' >&2".format(i))
            j = self.get_next_arm()
            self.run_arm(j)
        return
    
    def get_data(self):
        return pickle.load(open("mydata.p", "rb"))
