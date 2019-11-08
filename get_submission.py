#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:56:26 2019

@author: dine
"""

from Hyperband import HyperBand
from Worker_NEW import Worker
from ConfigGenerator import config_generator
from ConfigGenerator import discrete_uniform
from ConfigGenerator import log_uniform
import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
import time
import pickle 
import xgboost as xgb
from BAI import BAI
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") #Needed to save figures
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import pickle
import gc

params = {'max_depth': randint(3, 20),
              'subsample': uniform(0.5,0.5),
              'colsample_by_tree': uniform(0.5,0.5),
              'gamma': discrete_uniform(0, 0, 5, 0.3),
              'lambda':discrete_uniform(0,0,5,0.1),
              'alpha': discrete_uniform(0,0,5,0.3),
              'eta':log_uniform(-3,-0.301)
              }    
    
data = pickle.load(open("mydata.p", "rb"))
test = pickle.load(open("test.pkl", "rb"))
d_train = xgb.DMatrix(data[0], label=data[1])
ratio = float(np.sum(data[1] == 1)) / np.sum(data[1]==0)

for i in range(5):    
    for j in range(5):
    
        model = pickle.load(open("model_{}_{}".format(i,j), "rb"))
            
        t = [elem for elem in zip(list(params.keys()), model[0])]
        t.append(('scale_pos_weight',ratio))
        t.append(('eval_metric','auc'))
        t.append(('objective','binary:logistic'))
        t.append(('n_jobs',-1))
        print('Start training')
        bst = xgb.train(params=t, dtrain=d_train, num_boost_round=model[1], verbose_eval=True)
        print('Finished training')
        testmatrix = xgb.DMatrix(test)  
        print('Start prediction')
        y_pred = bst.predict(testmatrix)
        print('Saving prediction')
        submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred})
        submission.to_csv("submission_{}_{}.csv".format(i,j), index=False)
        
        
        