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

##### only random stuff in this file run cells individually


#%%

# example use of functions, Principal component analysis and model evaluations


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn import svm as SVM
from sklearn.naive_bayes import GaussianNB as NB


# custom imports
from funcs import plot_cv_confidence_vs_profit, score_dmc_profit,dmc_profit,cv_preds_and_confusion_matrix,cv_profits_for_models
from customClassifiers import CustomModelWithThreshold,PerceptronLearner



train = pd.read_csv('train.csv' ,delimiter="|")
test = pd.read_csv('test.csv', delimiter="|")
y = train.pop('fraud')
# example cross validation with the dmc19 custom score function
lr = LogisticRegression(C=10, solver='lbfgs')
cv_scores = cross_val_score(lr,train,y, cv=5, scoring=score_dmc_profit)
print("CV profit: " +   str(sum(cv_scores)))

# example use of the dmc_profit function
lr.fit(train,y)
y_pred = lr.predict(train)
print("Training Profit: " + str(dmc_profit(y,y_pred)))


### add new feature and normalize
train['totalItems'] = train['scannedLineItemsPerSecond'] * train['totalScanTimeInSeconds']
cols = train.columns
scaler = StandardScaler()
scaler.fit(train)
train = scaler.transform(train)

# plotting cross validation profits for different prediction thresholds
# you can pass any model
plot_cv_confidence_vs_profit(lr, train, y, 5, "LogisticReg")
rf = RandomForestClassifier()
plot_cv_confidence_vs_profit(rf, train, y, 5, "RandomForest")


### example use of the cross validation confusion matrix function ####
# also prints the confusion matrix
# takes a prediction threshold
predresults = cv_preds_and_confusion_matrix(lr,train,y,cvfolds=5,threshold=0.5)


# plot the first two principal component scores and mark the fraudsters
pca = PCA(n_components=7)
xax = 0
yax = 1
train_tr = pca.fit(train).transform(train)
nofrauds = train_tr[y==0]
frauds = train_tr[y==1]
plt.figure()
plt.scatter(nofrauds[:,xax], nofrauds[:,yax], color="black", alpha=0.7)
plt.scatter(frauds[:,xax], frauds[:,yax], color="yellow")
#now mark the fraudsters in blue which the classifier could not find
fraudresults = predresults[y==1]
fn = frauds[(fraudresults["true"] == 1) & (fraudresults["cvpredict"] == 0)]
plt.scatter(fn[:,xax], fn[:,yax], color="blue")
#mark the false positives in red
nonfrauds = train_tr[y==0]
nonfraudresults = predresults[y==0]
fp = nonfrauds[(nonfraudresults["true"] == 0) & (nonfraudresults["cvpredict"] == 1)]
plt.scatter(fp[:,xax], fp[:,yax], color="red")
plt.title("1,2PCA scores; yellow:fraud; blue:FN; red: FP")

print("")
print("")
# check some models and compare them with respect to F1, Acc and Profit
models = [LogisticRegression(C=10, solver='lbfgs'), SVM.SVC(gamma='auto'), DT(), KNN(5), NB()]
#uses the profit provided by the teachers
cv_profits_for_models(models,train,y)



#profit for the perceptron learner


perc = PerceptronLearner(1000)
cv_profits_for_models([perc], train,y)




#%%


## Pocket ALgorithm Prototype

import random


train = pd.read_csv('train.csv',delimiter="|")
train['totalItems'] = train['scannedLineItemsPerSecond'] * train['totalScanTimeInSeconds']
test = pd.read_csv('test.csv',delimiter="|")
y = train.pop('fraud')
### add new feature and normalize
train['totalItems'] = train['scannedLineItemsPerSecond'] * train['totalScanTimeInSeconds']
cols = train.columns
scaler = StandardScaler()
scaler.fit(train)
train = scaler.transform(train)
#
## perceptron learning
X0 = train.copy()

w0 = None
maxepoch = 2000

X = np.zeros((X0.shape[0], X0.shape[1]+1))

## add intercept (bias)
X[:,0] = 1
X[:,1:] = X0

N, D = X.shape
if w0 is None:        # initial weight vector
   w0 = np.zeros(D)
w = w0                # current weight vector

y_true = y.copy()
y_true[y_true==0] = -1
epochcounter = 0


profit_matrix = {(-1,-1): 0, (-1,1): -5, (1,-1): -25, (1,1): 5}
profits = []
weight_sequence = []

## note: we keep profits over epochs
# # and only reset profit when we hit an error
current_profit = 0
while(epochcounter<maxepoch):
    print(epochcounter)
    indexes = list(np.random.choice(N,N))
    epochcounter +=1
    instcounter = 0
    while(instcounter <= N):
        # draw instance randomly
        idx = random.choice(range(0,N,1))
        instcounter += 1
        ip = sum(X[idx]*w)
        # make prediction for the current instance
        if ip>=0:
            y_hat_i = 1
        else:
            y_hat_i = -1
        y_true_i = y_true[idx]
        current_profit = current_profit + profit_matrix[(y_hat_i,y_true_i)]
        # update weights in a perceptron learning fashion
        # same weights for all errors here
        if (y_hat_i != y_true_i):
             profits.append(current_profit)
             current_profit = 0
             weight_sequence.append(w)
             w = w + y_true_i * X[idx]


#%%


# feature distributions
             
             
             
train = pd.read_csv('train.csv',delimiter="|")
train['totalItems'] = train['scannedLineItemsPerSecond'] * train['totalScanTimeInSeconds']
test = pd.read_csv('test.csv',delimiter="|")

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



#%%



