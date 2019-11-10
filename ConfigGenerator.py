import pandas as pd
import numpy as np
from scipy.stats import uniform, randint

class config_generator(object):

    def __init__(self, params):
        if(type(params)!=type({})):
            raise ValueError('params must be a dict!')
        self.params = params
        self.config_counter = 0 #  THINK ABOUT MAKING STATIC
        
    def sample(self):
        config = []
        self.config_counter += 1
        for elem in self.params.values():
            config.append(elem.rvs())
        return config

    def sample_n(self, n):
        many_configs = []
        ids = []
        for _ in range(int(n)):
            many_configs.append(self.sample())
            ids.append(self.config_counter)
        return pd.DataFrame(data = {'conf':many_configs, 
                                    'L':[None]*int(n),
                                    'Id':ids})

class discrete_uniform(object):
    
    def __init__(self, disc_value, mi, rang, p):
        self.disc = disc_value
        self.min = mi
        self.range = rang
        self.p = p

    def rvs(self):
        u = uniform(0,1).rvs()
        if u < self.p:
            return self.disc
        else:
            return uniform(self.min, self.range).rvs()
	    
 
class log_uniform():        
    def __init__(self, a=-1, b=0, base=10):
        self.loc = a
        self.scale = b - a
        self.base = base

    def rvs(self, size=None, random_state=None):
        unif = uniform(loc=self.loc, scale=self.scale)
        if size is None:
            return np.power(self.base, unif.rvs(random_state=random_state))
        else:
            return np.power(self.base, unif.rvs(size=size, random_state=random_state))    

class log_int():
        
    def rvs(self):
        a = randint.rvs(1,4)
        if a == 1:
            return randint.rvs(2,4)
        elif a == 2:
            return randint.rvs(4,8)
        else:
           return randint.rvs(8,16) 