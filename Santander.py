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

training = pd.read_csv(r"C:\Users\dinello\Downloads\train.csv", index_col=0)
test = pd.read_csv(r"C:\Users\dinello\Downloads\test.csv", index_col=0)

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

X_train, X_test, y_train, y_test = train_test_split(X_sel, y, random_state=1301, stratify=y, test_size=0.4)

X_train

ratio = float(np.sum(y == 1)) / np.sum(y==0)

clf = xgb.XGBClassifier(missing=9999999999,
                max_depth = 5,
                n_estimators=1000,
                learning_rate=0.01, 
                nthread=-1,
                subsample=1.0,
                colsample_bytree=0.5,
                min_child_weight = 3,
                scale_pos_weight = ratio,
                reg_alpha=0.03,
                seed=1301)
                
clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="rmse",
        eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)