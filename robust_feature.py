#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:04:27 2019

@author: html
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import itertools
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from sklearn import svm as SVM
from funcs import cv_profits_for_models, cv_preds_and_confusion_matrix, CustomModelWithThreshold
from funcs import profit_scorer, profit_scoring
from customClassifiers import OutlierRemover
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization

from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import chi2



X_train = pd.read_csv('train.csv' ,delimiter="|")
X_test = pd.read_csv('test.csv', delimiter="|")

X_train = X_train[X_train['trustLevel']==2]
X_train.pop('trustLevel')
y = X_train.pop('fraud')
# leave this out and do everything plain
X_train['scannedLineItemsTotal'] = X_train['scannedLineItemsPerSecond'] * X_train['totalScanTimeInSeconds']
X_train['valuePerLineItem'] = X_train['grandTotal'] * X_train['scannedLineItemsTotal']
X_train['quantityModificationsPerLineItem'] = X_train['quantityModifications'] * X_train['scannedLineItemsTotal']
X_train['lineItemVoids*scansWithoutRegistration'] = X_train['lineItemVoids'] * X_train['scansWithoutRegistration']



#for var in list(X_train.columns):
#    X_train[str(var) + '_log'] = np.log(X_train[var])
    



X_train = np.float64(X_train)



# normalize data first to prevent 0's in dataset
prep_pipeline = Pipeline([
    ('scaling', StandardScaler())
])
X_train = prep_pipeline.fit_transform(X_train)


# generate features and rescale
prep_pipeline = Pipeline([
    ('interaction', PolynomialFeatures(3, interaction_only=False)),
    ('scaling', StandardScaler()),
])
X_train_all = prep_pipeline.fit_transform(X_train)

#remove the first var because it is the constant term
X_train_all = X_train_all[:,1:]


# obtain feature importance by xgboost
xgb = XGBClassifier(num_estimator=100)
xgb.fit(X_train_all, y)
imp = xgb.feature_importances_

lr = LogisticRegression(C=20, solver='lbfgs')
lr.fit(X_train_all, y)
imp = lr.coef_[0]



# order the feature indices by importance
imp = pd.DataFrame(imp)
imp = imp.sort_values(by=0, ascending=False)


# choose model
model = CustomModelWithThreshold(LogisticRegression(C=10, solver='lbfgs', max_iter=300), 0.9)

cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=False)
features_to_use = []
last_score = -10000
# add most important feature
X_temporary = pd.DataFrame(X_train_all[:,(list(imp.index))[0]])

# iteratively add features one by one
for featnum in (list(imp.index))[1:]:
    X_check = pd.concat([X_temporary,pd.Series(X_train_all[:,featnum])], axis=1)
    score = sum(cross_validate(model,X_check, y, scoring=profit_scoring, cv=cv)['test_score'])
    # add the feature ultimatively if score improved
    if score > last_score:
        X_temporary = pd.concat([X_temporary,pd.Series(X_train_all[:,featnum])], axis=1)
        features_to_use.append(featnum)
        last_score = score    
    print(last_score)    


def evaluateLogReg(C, pred_threshold):
    clf = LogisticRegression(C=C, solver='lbfgs', max_iter=10000)
    clf = CustomModelWithThreshold(clf,threshold=pred_threshold)
    return sum(cross_validate(clf,X_temporary, y, scoring=profit_scoring, cv=cv)['test_score'])


params_logreg = {
    'C': (0.001, 50),
    'pred_threshold': (0, 1)
}


optimization_logreg = BayesianOptimization(evaluateLogReg, params_logreg)
optimization_logreg.maximize(n_iter=1000, init_points=1000)








#%%