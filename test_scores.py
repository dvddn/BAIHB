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

def main():
    
    params = {'max_depth': randint(3, 20),
              'subsample': uniform(0.5,0.5),
              'colsample_by_tree': uniform(0.5,0.5),
              'gamma': discrete_uniform(0, 0, 5, 0.3),
              'lambda':discrete_uniform(0,0,5,0.1),
              'alpha': discrete_uniform(0,0,5,0.3),
              'eta':log_uniform(-3,-0.301)
              }    
    
    files = ['results.pkl','results_RS.pkl']
    data = pickle.load(open("mydata.p", "rb"))
    d_train = xgb.DMatrix(data[0], label=data[1])   
    ratio = float(np.sum(data[1] == 1)) / np.sum(data[1]==0)
    for i,name in enumerate(files):
            
        with open(name, 'rb') as handle:
            res = pickle.load(handle)
        try:
            models = res.sort_values('L')[-5:].copy()
        except:
            models = res.evals.sort_values('L')[-5:].copy()
        del res
        gc.collect()
        
        models['test'] = 0
        models.reset_index(drop=True, inplace=True)
        
#        with open('test.pkl', 'rb') as f:
#            test = pickle.load(f)
#          d_test = xgb.DMatrix(test)
        for index, row in enumerate(models.conf):
            t = [elem for elem in zip(list(params.keys()), row)]
            t.append(('scale_pos_weight',ratio))
            t.append(('eval_metric','auc'))
            t.append(('objective','binary:logistic'))
            t.append(('n_jobs',-1))
            bst = xgb.cv(params=t, dtrain=d_train, num_boost_round=4000, 
                                early_stopping_rounds=800, 
                                verbose_eval=True)
    #        nr = bst.iloc[z
            nr = bst.index[-1]
            exitname = 'model_{}_{}'.format(i+1,index)
            with open(exitname, 'wb') as handle:
                pickle.dump([row,nr], handle, protocol=pickle.HIGHEST_PROTOCOL)    
            
    
    

#    y_pred = clf.predict_proba(sel_test, ntree_limit=clf.best_iteration)
#
#    submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
#    submission.to_csv("submission.csv", index=False)

if __name__ == '__main__':
    main()
