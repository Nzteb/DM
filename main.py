#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn import svm as SVM
from sklearn.naive_bayes import GaussianNB as NB
from xgboost import XGBClassifier

# custom imports
from funcs import plot_cv_confidence_vs_profit, score_dmc_profit,dmc_profit,cv_preds_and_confusion_matrix,cv_profits_for_models, profit_scoring
from customClassifiers import CustomModelWithThreshold, TrustHard, PerceptronLearner
from pipes import CustomAttributeAdder,Scaling,RandomAttributeAdder,Transformer,ClfSwitcher

from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import VotingClassifier

# use sklearn pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import itertools



train = pd.read_csv('train.csv' ,delimiter="|")
test = pd.read_csv('test.csv', delimiter="|")
y = train.pop('fraud')




featureGeneration_pipeline = Pipeline([
    ("attribs_adder", CustomAttributeAdder()),    # returns pd.dataframe
    ("RandomAttributeAdder", RandomAttributeAdder())
    ])


preprocessing_pipeline = Pipeline([
    ("transformer", Transformer()),                # This class is still void
    ("scaler", Scaling(strategy='Standard')),
])


model_training_pipeline = Pipeline([
    ('feature_generation', featureGeneration_pipeline),
    ('preprocessing', preprocessing_pipeline),
    ('classifier', ClfSwitcher())
])



models  = [
('lr',LogisticRegression()),
('svm',SVM.SVC()),
#TrustHard(LogisticRegression(C=10)),
('perc',PerceptronLearner(100)),
('sgd',SGDClassifier()),
('xgb',XGBClassifier())
]



weights = list(itertools.product([0,1], repeat=len(models)))
weights.remove(tuple(np.zeros(len(models))))



parameters = [
    #{
    #   'classifier__estimator': [SGDClassifier()],
    #    'classifier__estimator__penalty': ('l2', 'elasticnet', 'l1'),
    #   'classifier__estimator__max_iter': [50, 80],
    #    'classifier__estimator__tol': [1e-4],
    #   'classifier__estimator__loss': ['hinge', 'modified_huber']
    #},
    
    #{
    #   'classifier__estimator': [LogisticRegression()],
    #    'classifier__estimator__C': [0.5,1,2,5,10,20,30],
    #},
    
    #{  
       #try different feature combinations  
    #   'feature_generation__attribs_adder__featurelist': [
    #                                  ['valuePerLineItem','quantityModificationsPerLineItem'],
    #                                  ['quantityModificationsPerLineItem'],
    #                                  ['valuePerLineItem']],  
       {'classifier__estimator': [VotingClassifier(estimators=models,voting='hard')],
       'classifier__estimator__weights': weights,
       # params for the single models
       
       'classifier__estimator__lr__C': [5,10,20,30,40,50,100,200],
       
       'classifier__estimator__sgd__loss':['modified_huber'],
       'classifier__estimator__sgd__max_iter':[100],
       'classifier__estimator__sgd__penalty':['l2', 'elasticnet', 'l1'],
       
       
       
       'classifier__estimator__xgb__max_depth': list(range(10)),
       'classifier__estimator__xgb__n_estimators': [10,15,30,70,200],
       
       'classifier__estimator__svm__C': list(np.arange(0.1,5,0.2)),
       'classifier__estimator__svm__kernel': ['linear', 'poly', 'rbf']
       
    }
    
    
    #{
    #    'classifier__estimator': [XGBClassifier()],
    #    'classifier__estimator__n_estimators': [50, 100, 150],
    #    'classifier__estimator__reg_alpha': [0, 0.05, 0.1]
    #},
    #{
    #    'classifier__estimator': [RandomForestClassifier()],
    #    'classifier__estimator__min_samples_split': [2, 4, 6],
    #    'classifier__estimator__criterion': ['gini', 'entropy']
    #}
    
    
    
]

train['fraud'] = y
train1 = train[train["trustLevel"]==2]
train2 = train[train["trustLevel"]==1]

y1 = train1.pop('fraud')
y2 = train2.pop('fraud')


cv = StratifiedKFold(n_splits=10, random_state=42)
gscv = GridSearchCV(model_training_pipeline, parameters, cv=cv, n_jobs=-1, scoring=profit_scoring, verbose=3)
gscv.fit(train1, y1)
print(gscv.best_score_)
print(gscv.best_params_)
gscv.best_estimator_.named_steps



















