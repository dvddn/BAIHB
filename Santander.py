# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 13:36:07 2019

@author: dinello
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") #Needed to save figures
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import pickle

training = pd.read_csv("/home/dine/Downloads/train.csv", index_col=0)
test = pd.read_csv("/home/dine/Downloads/test.csv", index_col=0)

print(training.shape)
print(test.shape)


training = training.replace(-999999,2)



X = training.iloc[:,:-1]
y = training.TARGET

X['n0'] = (X == 0).sum(axis=1)

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

X_normalized = normalize(X, axis=0)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)
X['PCA1'] = X_pca[:,0]
X['PCA2'] = X_pca[:,1]

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

p = 75 # 261 features validation_1-auc:0.848642


X_bin = Binarizer().fit_transform(scale(X))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)

chi2_selected = selectChi2.get_support()
chi2_selected_features = [ f for i,f in enumerate(X.columns) if chi2_selected[i]]
print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
   chi2_selected_features))
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
   f_classif_selected_features))
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
features = [ f for f,s in zip(X.columns, selected) if s]
print (features)

X_sel = X[features]

data = [X_sel, y]

#with open('/home/dine/Documents/Python_code/BAIHB/BAIHB/mydata.p', 'wb') as handle:
#    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

test['n0'] = (test == 0).sum(axis=1)
# test['logvar38'] = test['var38'].map(np.log1p)
# # Encode var36 as category
# test['var36'] = test['var36'].astype('category')
# test = pd.get_dummies(test)
test_normalized = normalize(test, axis=0)
pca = PCA(n_components=2)
test_pca = pca.fit_transform(test_normalized)
test['PCA1'] = test_pca[:,0]
test['PCA2'] = test_pca[:,1]
sel_test = test[features]    

with open('/home/dine/Documents/Python_code/BAIHB/BAIHB/test.pkl', 'wb') as handle:
    pickle.dump(sel_test, handle, protocol=pickle.HIGHEST_PROTOCOL)


#y_pred = clf.predict_proba(sel_test, ntree_limit=clf.best_iteration)

#submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
#submission.to_csv("submission.csv", index=False)
