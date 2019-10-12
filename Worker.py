import pandas as pd
import xgboost as xgb
import numpy as np
import pickle

class Worker(object):
    def __init__(self, date, params):
        self.data = pd.read_pickle("PATH")
        self.date = date
        self.params = params
        return

    def set_date(self, date):
        self.date = date
        return

    def eval(self, config_vals, d, r, id):
        t = [elem for elem in zip(list(self.params.keys()),
                                  config_vals)]
        cv_data = self.get_training(self.date)
        scores = []
        for split in [0.7, 0.75, 0.8]:
            index_train, index_val = self.get_stocks_quintile(cv_data, split)
            cond_train = cv.data.date.apply(lambda x: x in index_train)
            cond_val =  ~cond_train
            train = cv_data[cond_train]
            val = cv_data[cond_val]
            res = self.run(t, train, val, d, r, id=id)
            scores.append(res)
            print('Training XGB with conf {} on {} split and budget {}: {}'.format(
                id, split,r+d,res))
        return np.mean(scores)

    def run(self, t, train, val, d, r, return_round=False, id=-1):
        train = train[train.Label != 0]
        train_y = train['Label'].apply(lambda x: 0 if x == -1 else 1)
        train_x = train.drop(['Label','Label2','stock','date'], axis=1)
        val_y = val['Label2'].apply(lambda x: 0 if x == -1 else 1)
        val_x = val.drop(['Label','Label2','stock','date']), axis=1)
        d_train = xgb.DMatrix(train_x, label=train_y)
        d_val = xgb.DMatrix(val_x, label=val_y)
        evallist = [(d_train, 'train'), (d_val,'eval')]
        t.append(('metric','merror'))
        t.append(('objective','binary:logistic'))
        t.append(('n_jobs',-1))
        early = None
        if (r+d)*100 > 1000:
            early = int(r+d)*100/2
        try:
            with open('PATH/model_{}_{}.pickle'.format(id, int(r)), 'rb') as f:
                bst = pickle.load(f)
            bst = xgb.train(params=t, dtrain=d_train, num_boost_round=int(d)*100, 
                            evals=evallist, early_stopping_rounds=early, 
                            verbose_eval=False, xgb_model=bst)
        except:
            bst = xgb.train(params=t, dtrain=d_train, num_boost_round=int(r+d)*100, 
                            evals=evallist, early_stopping_rounds=early, 
                            verbose_eval=False)
            best_iter = bst.attr('best_iteration')
            if best_iter is None:
                best_iter = int(r+d)*100
        if id!=-1:
            with open('PATH/model_{}_{}'.format(id, int(r+d)), 'wb') as handle:
                pickle.dump(bst, handle, protocol=pickle.HIGHEST_PROTOCOL)
        preds = bst.predict(d_val) - 0.5
        results = pd.DataFrame(data={'preds':preds, 'truth':val_y.apply(
                lambda x: -1 if x==0 else 1)})
        results['acc'] = results.preds*results.truth
        if return_round:
            return (results.acc>0).mean(), best_iter
        else:
            return (results.acc>0).mean()

    def get_training(self, date):
        """returns training data two years before and six years after a given date"""
        y = self.data[((self.data.date < date - 20000) | (self.data.date > date + 60000))]
        return y.drop(['R1M_shifted', 'INDUSTRY_SECTOR'], axis=1)

    def get_validation(self, date):
        """returns training data two years before a given date"""
        y = self.data[((self.data.date > date - 20000) & (self.data.date < date))]
        return y.drop(['R1M_shifted', 'INDUSTRY_SECTOR'], axis=1)

    def get_training(self, date):
        """returns training data two years before and six years after a given date"""
        y = self.data[((self.data.date < date - 20000) | (self.data.date > date + 60000))]
        return y.drop(['R1M_shifted', 'INDUSTRY_SECTOR'], axis=1)


