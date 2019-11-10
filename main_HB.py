#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 01:27:18 2019

@author: dine
"""

from Hyperband import HyperBand
from Worker_NEW import Worker
from ConfigGenerator import config_generator
from ConfigGenerator import discrete_uniform
from ConfigGenerator import log_uniform, log_int
import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
import time
import pickle 
import xgboost as xgb
from BAI import BAI
import datetime

def main():
    a=datetime.datetime.now()
    data = pickle.load(open("mydata.p", "rb"))
    params = {'max_depth':log_int(),
              'subsample': uniform(0.5,0.5),
              'colsample_by_tree': uniform(0.5,0.5),
              'gamma': discrete_uniform(0, 0, 5, 0.3),
              'lambda':discrete_uniform(0,0,5,0.1),
              'alpha': discrete_uniform(0,0,5,0.3),
              'eta':log_uniform(-2.3,-0.5228)
              }
    eta_interval = [0.005,0.3]
 
    model = Worker(params, data)
    hb = HyperBand(model, config_generator(params), R=81)
    hb.run_n(25)
#    print(hb.evals)
#    baihb = BAI(3,params,eta_interval,81,3.0)
#    baihb.run_n(50)
#    
    tab = hb.evals
#    for arm in baihb.arms:
#        tab = pd.concat([tab,arm.hb.evals])
#    print(tab)
    with open('results_HB_new3.py', 'wb') as handle:
        pickle.dump(tab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(datetime.datetime.now() - a)    
if __name__ == '__main__':
    main()
