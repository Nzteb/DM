#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from funcs import plot_cv_confidence_vs_profit, dmc_profit,score_dmc_profit
from customClassifiers import CustomModelWithThreshold

import itertools



##### only random stuff in this file


train = pd.read_csv('train.csv',delimiter="|")
train['totalItems'] = train['scannedLineItemsPerSecond'] * train['totalScanTimeInSeconds']


print(train.columns)
print("")
print("")

print("distribution of trust level in the train data")
for i in range(6):
    print("trust level: " + str(i+1))
    print( sum(train["trustLevel"] == i+1))
    
print("distribution of trust level in the test data")  
for i in range(6):
   print("trust level: " + str(i+1))
   print( sum(test["trustLevel"] == i+1))   
   
print("distribution of frauds in the trust levels")   
for i in range(6):
    lvl = train[train["trustLevel"] == (i+1)]
    num = sum(lvl["fraud"] == 1) 
    print("num frauds in trust level" + ": " + str(i+1) + " " +  str(num))   


corr = train.corr()

y = train.pop('fraud')
test = pd.read_csv('test.csv', delimiter="|")
scaler = StandardScaler()
scaler.fit(train)
train = scaler.transform(train)

lr = LR()





print(dmc_profit(cvres['true'], cvres['cvpredict']))
print(sum(cross_val_score(lr, train, y, cv=5, scoring=score_dmc_profit)))










lr_raw = LR()
lr =  CustomModelWithThreshold(lr_raw,0.62)
lr.fit(train,y)

y_pred = lr.predict(train)
probabs = lr.predict_proba(train)


prds = pd.DataFrame()
prds['true'] = y
prds['pred'] = y_pred
prds['probab'] = probabs[:,1]




#score_dmc_profit = make_scorer(dmc_profit, greater_is_better=True)
print(dmc_profit(y, y_pred)) 
print(score_dmc_profit(lr,train, y))     
#print(np.mean(cross_val_score(lr, train, y, cv=5, scoring=score_dmc_profit)))
#rf = RF()
plot_cv_confidence_vs_profit(lr, train, y,5, "logisticReg")
#plot_cv_confidence_vs_profit(rf, train, y,5, "RandomForest")

plot_cv_confidence_vs_profit(lr, train.loc[:,["trustLevel","totalItems"]], y,5, "logisticReg")
     



#calculate all logs and interaction terms
####read new
#train = pd.read_csv('train.csv',delimiter="|")
#y = train.pop("fraud")
#cols = train.columns
#for (avar, bvar) in itertools.combinations(cols,2):
#     train[avar + "*" + bvar] = train[avar] * train[bvar]
#
#for avar in cols:
#    if (sum(train[avar]==0)==0):
#        train[avar + "_log"] = np.log(train[avar])
# 
#
#               
#corr = pd.concat([train,y] ,axis=1).corr()
#corr = corr["fraud"]
#corr = corr[:-1]
#corr = np.abs(corr)
#corr = corr.sort_values(ascending=False) 
#corr = corr[corr > 0.1]
#corr = corr[1:]
#newdf = pd.DataFrame()
#
#for var in corr.index:
#    newdf[var] = train[var]


#plot_cv_confidence_vs_profit(lr, newdf, y,5, "logisticReg")

scaler = StandardScaler()
scaler.fit(train)
train = scaler.transform(train)


pca = PCA(n_components=15)
train_tr = pca.fit(train).transform(train)

nofrauds = train_tr[y==0]
frauds = train_tr[y==1]
plt.scatter(nofrauds[:,0], nofrauds[:,1], color="black", alpha=0.7)
plt.scatter(frauds[:,0], frauds[:,1], color="yellow") 

    
plot_cv_confidence_vs_profit(lr,train_tr, y,5, "logisticReg on principal comp")

lr.fit(train_tr, y)
preds = lr.predict(train_tr)
     








