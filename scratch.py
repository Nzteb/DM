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


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import itertools

from sklearn import svm as SVM
from funcs import cv_profits_for_models, cv_preds_and_confusion_matrix, CustomModelWithThreshold
from funcs import profit_scorer, profit_scoring
from customClassifiers import OutlierRemover
from xgboost import XGBClassifier

train = pd.read_csv('train.csv' ,delimiter="|")
test = pd.read_csv('test.csv', delimiter="|")


train['scannedLineItemsTotal'] = train['scannedLineItemsPerSecond'] * train['totalScanTimeInSeconds']
train['valuePerLineItem'] = train['grandTotal'] / train['scannedLineItemsTotal']
train['quantityModificationsPerLineItem'] = train['quantityModifications'] / train['scannedLineItemsTotal']
train['lineItemVoids*scansWithoutRegistration'] = train['lineItemVoids'] * train['scansWithoutRegistration']

train = train[train['trustLevel']]
y = train.pop('fraud')
cols = train.columns


scaler = StandardScaler()
scaler.fit(train)
train = scaler.transform(train)

train = pd.DataFrame(train, columns=cols)
#train['fraud']=np.float64(y) 


om = OutlierRemover(LogisticRegression(C=20))
#res = cv_preds_and_confusion_matrix(om, train,y, cvfolds=10, probs=False)

xgb = XGBClassifier(n_estimators=331, max_depth=8, gamma=0.03, reg_alpha=0.78)

cv = StratifiedKFold(n_splits=10, random_state=42)
print(sum(cross_validate(om, train,y, cv=cv, scoring=profit_scoring )['test_score']))

#res = cv_preds_and_confusion_matrix(LogisticRegression(C=20), train,y, cvfolds=10, probs=False)




#%% outlier plots


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import itertools

from sklearn import svm as SVM
from funcs import cv_profits_for_models, cv_preds_and_confusion_matrix, CustomModelWithThreshold
from funcs import profit_scorer
from customClassifiers import OutlierRemover


train = pd.read_csv('train.csv' ,delimiter="|")
test = pd.read_csv('test.csv', delimiter="|")


train['scannedLineItemsTotal'] = train['scannedLineItemsPerSecond'] * train['totalScanTimeInSeconds']
train['valuePerLineItem'] = train['grandTotal'] / train['scannedLineItemsTotal']
train['quantityModificationsPerLineItem'] = train['quantityModifications'] / train['scannedLineItemsTotal']
train['lineItemVoids*scansWithoutRegistration'] = train['lineItemVoids'] * train['scansWithoutRegistration']

train = train[train['trustLevel']<2]
y = train.pop('fraud')
cols = train.columns


scaler = StandardScaler()
scaler.fit(train)
train = scaler.transform(train)

train = pd.DataFrame(train, columns=cols)
train['fraud']=np.float64(y) 



#interesting plots
pairs=[#('trustLevel', 'scannedLineItemsPerSecond'),
        #('trustLevel', 'scannedLineItemsTotal'),
        #('trustLevel', 'valuePerLineItem'),
        #('trustLevel', 'quantityModificationsPerLineItem'),
        ('totalScanTimeInSeconds', 'scannedLineItemsPerSecond'),
        #('totalScanTimeInSeconds', 'scannedLineItemsTotal'),
        #('totalScanTimeInSeconds', 'valuePerLineItem'),
        ('grandTotal', 'scannedLineItemsPerSecond'),
        ('grandTotal', 'valuePerSecond'),
        #('grandTotal', 'lineItemVoidsPerPosition'),
        #('grandTotal', 'scannedLineItemsTotal'),
        #('grandTotal', 'valuePerLineItem'),
        #('grandTotal', 'quantityModificationsPerLineItem'),
        #('lineItemVoids', 'scannedLineItemsPerSecond'),
        #('lineItemVoids', 'valuePerSecond'),
        #('lineItemVoids', 'scannedLineItemsTotal'),
        #('lineItemVoids', 'quantityModificationsPerLineItem'),
        #('scansWithoutRegistration', 'scannedLineItemsPerSecond'),
        #('scansWithoutRegistration', 'lineItemVoidsPerPosition'),
        #('scansWithoutRegistration', 'quantityModificationsPerLineItem'),
        #('quantityModifications', 'scannedLineItemsTotal'),
        ('scannedLineItemsPerSecond', 'valuePerSecond'),
        #('scannedLineItemsPerSecond', 'scannedLineItemsTotal'),
        #('scannedLineItemsPerSecond', 'valuePerLineItem'),
        #('valuePerLineItem', 'quantityModificationsPerLineItem')]
        ('grandTotal', 'valuePerSecond')]
    



for pair in pairs:
    train_0 = train[train['fraud']==0]
    train_1 = train[train['fraud']==1]
    
    #use axis scaling of the fraudster instances
    minx = min(train_1[pair[0]])
    maxx = max(train_1[pair[0]])
    
    miny = min(train_1[pair[1]])
    maxy = max(train_1[pair[1]])
    
    
    plt.xlabel(pair[0])
    plt.ylabel(pair[1])
    plt.xlim(minx-0.1,maxx+0.1)
    plt.ylim(miny-0.1,maxy+0.1)
    plt.scatter(train_0[pair[0]], train_0[pair[1]])
    plt.scatter(train_1[pair[0]], train_1[pair[1]], color='yellow')
    plt.show()


y = train.pop('fraud')
lr = LogisticRegression(C=20)


res = cv_preds_and_confusion_matrix(lr,train,y, cvfolds=10)


print(len(train))
train['fraud'] = y


loose1 = -1*((train['valuePerSecond']>-0.1) & (train['fraud']==1.0)) + 1
loose2 = -1*((train['scannedLineItemsPerSecond']>0.2) & (train['fraud']==1.0)) +1

train = train[loose1.astype('bool')]
train = train[loose2.astype('bool')]

print(len(train))

y = train.pop('fraud')

res = cv_preds_and_confusion_matrix(lr, train,y, cvfolds=10)








#%%


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import itertools

from sklearn import svm as SVM
from funcs import cv_profits_for_models, cv_preds_and_confusion_matrix, CustomModelWithThreshold
from funcs import profit_scorer





train = pd.read_csv('train.csv' ,delimiter="|")
test = pd.read_csv('test.csv', delimiter="|")

train = train[train['trustLevel']<3]

y = train.pop('fraud')
cols = train.columns
scaler = StandardScaler()
scaler.fit(train)
train = scaler.transform(train)

train = pd.DataFrame(train, columns=cols)
train['fraud']=y 



train1 = train
#train1 = train[train['trustLevel']==1]
train2 = train[train['trustLevel']==2]

y1 = train1.pop('fraud')
train1['scannedLineItemsTotal'] = train1['scannedLineItemsPerSecond'] * train1['totalScanTimeInSeconds']
train1['valuePerLineItem'] = train1['grandTotal'] / train1['scannedLineItemsTotal']
train1['quantityModificationsPerLineItem'] = train1['quantityModifications'] / train1['scannedLineItemsTotal']
train1['lineItemVoids*scansWithoutRegistration'] = train1['lineItemVoids'] * train1['scansWithoutRegistration']



y2 = train2.pop('fraud')
train2['scannedLineItemsTotal'] = train2['scannedLineItemsPerSecond'] * train2['totalScanTimeInSeconds']
train2['valuePerLineItem'] = train2['grandTotal'] / train2['scannedLineItemsTotal']
train2['quantityModificationsPerLineItem'] = train2['quantityModifications'] / train2['scannedLineItemsTotal']
train2['lineItemVoids*scansWithoutRegistration'] = train2['lineItemVoids'] * train2['scansWithoutRegistration']




pairs = list(itertools.combinations(train1.columns,2))


train1['fraud']=y1
train2['fraud']=y2

train1.sort_values(by='fraud',ascending=False)

for pair in pairs:
    print(pair)
    train1_0 = train1[train1['fraud']==0]
    train1_1 = train1[train1['fraud']==1]
    plt.scatter(train1_0[pair[0]], train1_0[pair[1]])
    plt.scatter(train1_1[pair[0]], train1_1[pair[1]], color='yellow')
    plt.show()
    train2_0 = train2[train2['fraud']==0]
    train2_1 = train2[train2['fraud']==1]
    plt.scatter(train2_0[pair[0]], train2_0[pair[1]])
    plt.scatter(train2_1[pair[0]], train2_1[pair[1]], color='yellow')
    plt.show()




#interesting plots
pairs=[('trustLevel', 'scannedLineItemsPerSecond'),
        ('trustLevel', 'scannedLineItemsTotal'),
        ('trustLevel', 'valuePerLineItem'),
        ('trustLevel', 'quantityModificationsPerLineItem'),
        ('totalScanTimeInSeconds', 'scannedLineItemsPerSecond'),
        ('totalScanTimeInSeconds', 'scannedLineItemsTotal'),
        ('totalScanTimeInSeconds', 'valuePerLineItem'),
        ('grandTotal', 'scannedLineItemsPerSecond'),
        ('grandTotal', 'valuePerSecond'),
        ('grandTotal', 'lineItemVoidsPerPosition'),
        ('grandTotal', 'scannedLineItemsTotal'),
        ('grandTotal', 'valuePerLineItem'),
        ('grandTotal', 'quantityModificationsPerLineItem'),
        ('lineItemVoids', 'scannedLineItemsPerSecond'),
        ('lineItemVoids', 'valuePerSecond'),
        ('lineItemVoids', 'scannedLineItemsTotal'),
        ('lineItemVoids', 'quantityModificationsPerLineItem'),
        ('scansWithoutRegistration', 'scannedLineItemsPerSecond'),
        ('scansWithoutRegistration', 'lineItemVoidsPerPosition'),
        ('scansWithoutRegistration', 'quantityModificationsPerLineItem'),
        ('quantityModifications', 'scannedLineItemsTotal'),
        ('scannedLineItemsPerSecond', 'valuePerSecond'),
        ('scannedLineItemsPerSecond', 'scannedLineItemsTotal'),
        ('scannedLineItemsPerSecond', 'valuePerLineItem'),
        ('valuePerLineItem', 'quantityModificationsPerLineItem')]
    
    
    



#%%


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import itertools

from sklearn import svm as SVM
from funcs import cv_profits_for_models, cv_preds_and_confusion_matrix, CustomModelWithThreshold
from funcs import profit_scorer

svm = SVM.SVC(C=10)

train = pd.read_csv('train.csv' ,delimiter="|")
train['scannedLineItemsTotal'] = train['scannedLineItemsPerSecond'] * train['totalScanTimeInSeconds']
train = train[train["trustLevel"]<3]
y = train.pop('fraud')


scaler = StandardScaler()
scaler.fit(train)
train_ = scaler.transform(train)
lr.fit(train_,y)

res = cv_preds_and_confusion_matrix(svm, train_,y,cvfolds=10, probs=False)
svm.fit(train_,y)
svm.predict(train)

profit_scorer(y,svm.predict(train_))


#%%

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import itertools

from funcs import cv_profits_for_models, cv_preds_and_confusion_matrix, CustomModelWithThreshold

train = pd.read_csv('train.csv' ,delimiter="|")

train = train[train["trustLevel"]<3]
#train.pop("trustLevel")

test = pd.read_csv('test.csv', delimiter="|")
y = train.pop('fraud')
train['scannedLineItemsTotal'] = train['scannedLineItemsPerSecond'] * train['totalScanTimeInSeconds']
#train['valuePerLineItem'] = train['grandTotal'] / train['scannedLineItemsTotal']
#train['quantityModificationsPerLineItem'] = train['quantityModifications'] / train['scannedLineItemsTotal']


cols = train.columns
lr = CustomModelWithThreshold(LogisticRegression(C=42), 0.5)
scaler = StandardScaler()
scaler.fit(train)
train_ = scaler.transform(train)
lr.fit(train_,y)

#coef = list(lr.coef_[0])
#
#for i in range(len(coef)):
#    print("var: {}".format((cols[i])))
#    print("coef: {}.".format(coef[i]))
#

res = cv_preds_and_confusion_matrix(lr, train_,y,cvfolds=10)




#%%  from the original variables + total line items test all possible feature subsets
# should be incorporated into the pipeline and adjust by the new features from niklas

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import itertools

from funcs import cv_profits_for_models, cv_preds_and_confusion_matrix

train = pd.read_csv('train.csv' ,delimiter="|")
test = pd.read_csv('test.csv', delimiter="|")
y = train.pop('fraud')
train['scannedLineItemsTotal'] = train['scannedLineItemsPerSecond'] * train['totalScanTimeInSeconds']
train['valuePerLineItem'] = train['grandTotal'] / train['scannedLineItemsTotal']

lr = LogisticRegression(C=20)
counter = 0
cols = train.columns
profits = []
feat_list = []


for size in range(1,len(cols)+1):
    features = list(itertools.combinations(cols,size))
    for comb in list(features):
        print(list(comb))
        train_ = train[list(comb)]
        scaler = StandardScaler()
        scaler.fit(train_)
        train_ = scaler.transform(train_)
        profit = cv_profits_for_models([lr], train_, y)
        profits.append(profit)
        feat_list.append(list(comb))
        counter += 1
        print("########################")
        print(counter)
        print("########################")



idx = profits.index(max(profits))
best = feat_list[idx]


print("Best features are:")
print(best)        
        

#%%

#investigate different linear models

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
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier


# custom imports
from funcs import plot_cv_confidence_vs_profit, score_dmc_profit,dmc_profit,cv_preds_and_confusion_matrix,cv_profits_for_models
from customClassifiers import CustomModelWithThreshold,PerceptronLearner, TrustHard, VoteClassifier



train = pd.read_csv('train.csv' ,delimiter="|")
test = pd.read_csv('test.csv', delimiter="|")
y = train.pop('fraud')

### add new feature and normalize
train['scannedLineItemsTotal'] = train['scannedLineItemsPerSecond'] * train['totalScanTimeInSeconds']
train['valuePerLineItem'] = train['grandTotal'] / train['scannedLineItemsTotal']
#train['quantityModificationsPerLineItem'] = train['quantityModifications'] / train['scannedLineItemsTotal']
cols = train.columns
scaler = StandardScaler()
scaler.fit(train)
train = scaler.transform(train)

cv = 10
res1 = cv_preds_and_confusion_matrix(LogisticRegression(C=10),train,y,cvfolds=cv, probs=False)
res2 = cv_preds_and_confusion_matrix(SVM.SVC(gamma='auto'), train,y,cvfolds=cv,probs=False)
res3 = cv_preds_and_confusion_matrix(TrustHard(LogisticRegression(C=10)), train,y,cvfolds=cv, probs=False)
res4 = cv_preds_and_confusion_matrix(PerceptronLearner(100),train,y,cvfolds=cv,probs=False)
res5 = cv_preds_and_confusion_matrix(SGDClassifier(loss='modified_huber', max_iter=50), train,y,cvfolds=cv,probs=False)
res6 = cv_preds_and_confusion_matrix(XGBClassifier(max_depth=4), train,y,cvfolds=10, probs=False)


for res in [res2,res3, res4, res5, res6]:
    res.pop("true")
    res1 = pd.concat((res1, res), axis=1)
    


train = pd.DataFrame(train, columns=cols)
res1 = pd.concat((res1,train), axis=1)


models  = [
LogisticRegression(C=10),
SVM.SVC(gamma='auto'),
#TrustHard(LogisticRegression(C=10)),
PerceptronLearner(100),
SGDClassifier(loss='modified_huber', max_iter=50),
XGBClassifier(max_depth=4)]


cv_preds_and_confusion_matrix(VoteClassifier(models,[1,0,1,0,1]),train,y, cvfolds=10, probs=False)








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



