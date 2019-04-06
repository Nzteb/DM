#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# custom imports
from funcs import plot_cv_confidence_vs_profit, score_dmc_profit,dmc_profit,cv_preds_and_confusion_matrix
from customClassifiers import CustomModelWithThreshold



train = pd.read_csv('train.csv' ,delimiter="|")
test = pd.read_csv('test.csv', delimiter="|")
y = train.pop('fraud')

# example cross validation with the dmc19 custom score function
lr = LogisticRegression()
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
predresults = cv_preds_and_confusion_matrix(lr,train,y,cvfolds=5,threshold=0.47)


# plot the first two principal component scores and mark the fraudsters
pca = PCA(n_components=5)
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








    


    