import math
import numpy as np
from ConfigGenerator import config_generator
import pandas as pd
import os

class HyperBand(object):

    def __init__(self, model, config_gen, R, eta=3.0):
        self.R = R
        self.eta = eta
        self.s_max = int(math.log(self.R, self.eta))
        self.B = (self.s_max + 1)*self.R
        self.model = model
        self.config_gen = config_gen
        self.evals = pd.DataFrame()
        print("HB object created!")

    def set_model(self, model):
        self.model = model
        return

    def run_n(self, n):
        print("Running HB {} times".format(n))
        for _ in np.arange(n):
            self.run()
        print(self.evals)
        return

    def run(self):
        print("Running HB")
        for s in np.arange(self.s_max, -1, -1):
            n = self.B*(self.eta**s)/(self.R*(s+1))//1
            r = self.R*self.eta**(-s)
            T = self.successive_halvings(n,r,s)
            os.popen('rm ./MODELS/*')
        return
    
    def successive_halvings(self, n, r, s):
        T = self.get_config(n)
        #print(T.conf)
        L = None
        r_i = 0
        for i in range(s+1):
            n_i = n * (self.eta ** (-i)) // 1
            old_r = r_i
            r_i = r * (self.eta ** i)
            d_i = r_i - old_r
            T.L = T.apply(lambda x: self.model.run(x.conf, d_i, old_r, False, x.Id), axis=1)
            if i < s:
                T = self.top_k(T, int(n_i/self.eta))
        self.evals = pd.concat([self.evals, T])
        return T

    def top_k(self, T, k):
        mysort = T.sort_values('L')
        return mysort.iloc[-k:].reset_index(drop=True)

    def get_config(self, n):
        return self.config_gen.sample_n(n)
