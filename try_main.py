from Hyperband import HyperBand
from Worker_NEW import Worker
from ConfigGenerator import config_generator
from ConfigGenerator import discrete_uniform
import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
import time
import pickle 
import xgboost as xgb
from BAI import BAI

def main():
    data = pickle.load(open("mydata.p", "rb"))
    params = {'max_depth': randint(3, 20),
              'subsample': uniform(0.5,0.5),
              'colsample_by_tree': uniform(0.5,0.5),
              'gamma': discrete_uniform(0, 0, 5, 0.3),
              'lambda':discrete_uniform(0,0,5,0.1),
              'alpha': discrete_uniform(0,0,5,0.3)}
    eta_interval = [0.001,0.5]
 
    #model = Worker(params, data)
    #hb = HyperBand(model, params, R=9)
    #hb.run_n(2)
    #print(hb.evals)
    baihb = BAI(5,params,eta_interval,3,3.0)
    baihb.run_n(2)
    with open('results_TS.pkl', 'wb') as handle:
        pickle.dump(baihb, handle, protocol=pickle.HIGHEST_PROTOCOL)
if __name__ == '__main__':
    main()