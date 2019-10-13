import pandas as pd
import xgboost as xgb
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score

class Worker(object):
    def __init__(self, params, data):
        self.params = params
        self.data = data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data[0], data[1], random_state=1301, stratify=data[1], test_size=0.4)
        self.ratio = float(np.sum(data[1] == 1)) / np.sum(data[1]==0)
        return

    def run(self, config_vals, d, r, return_round=False, Id=-1):
        t = [elem for elem in zip(list(self.params.keys()),
                                  config_vals)]
        d_train = xgb.DMatrix(self.X_train, label=self.y_train)
        d_val = xgb.DMatrix(self.X_test, label=self.y_test)
        evallist = [(d_train, 'train'), (d_val,'eval')]
        t.append(('scale_pos_weight',self.ratio))
        t.append(('eval_metric','aucpr'))
        t.append(('objective','binary:logistic'))
        t.append(('n_jobs',-1))
        early = None
        if (r+d)*30 > 200:
            early = int(r+d)*30/3
        try:
            with open('./MODELS/model_{}_{}.pickle'.format(Id, int(r)), 'rb') as f:
                bst = pickle.load(f)
            print('Running XGB, id =', Id, ' budget: ', int(d)*30 )
            bst = xgb.train(params=t, dtrain=d_train, num_boost_round=int(d)*30, 
                            evals=evallist, early_stopping_rounds=early, 
                            verbose_eval=False, xgb_model=bst)
        except:
            print('Running XGB, id =', Id, ' budget: ', int(r+d)*30 )
            bst = xgb.train(params=t, dtrain=d_train, num_boost_round=int(r+d)*30, 
                            evals=evallist, early_stopping_rounds=early, 
                            verbose_eval=False)
            best_iter = bst.attr('best_iteration')
            if best_iter is None:
                best_iter = int(r+d)*100
        if id!=-1:
            with open('./MODELS/model_{}_{}'.format(Id, int(r+d)), 'wb') as handle:
                pickle.dump(bst, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        preds = bst.predict(d_val)
        results = pd.DataFrame(data={'preds':preds, 'truth':self.y_test})
        acc = roc_auc_score(results.truth, results.preds)
        if return_round:
            return acc, best_iter
        else:
            return acc

    def get_training(self):
        return self.data[0]

    def get_validation(self):
        return self.data[1]



