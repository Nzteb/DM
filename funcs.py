#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.metrics import make_scorer
from customClassifiers import CustomModelWithThreshold
from sklearn.model_selection import cross_val_score,cross_val_predict, cross_validate
from sklearn.model_selection import StratifiedKFold
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd



cv = StratifiedKFold(n_splits=10, random_state=42)
# raw profit function provided by the teachers
# same as dmc_profit
def profit_scorer(y, y_pred):
    profit_matrix = {(0,0): 0, (0,1): -5, (1,0): -25, (1,1): 5}
    return sum(profit_matrix[(pred, actual)] for pred, actual in zip(y_pred, y))
#sklearn custom score function
profit_scoring = make_scorer(profit_scorer, greater_is_better=True)


# raw profit function of the competition
# takes true labels and predictions and returns profit
# (iterable,iterable) --> int
def dmc_profit(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = sum((y_true == 1) & (y_pred == 1))
    #TN = sum((cvres['true'] == 0) & (cvres['cvpredict'] == 0))
    FP = sum((y_true == 0) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))
    profit = 5*TP -25*FP -5*FN
    return profit

# sklearn custom score function defined by the raw dmc_profit
# e. g. can be used in cross validation 
# callable as: score_dmc_profit(sklearnmodel, X_val, y_true)    
score_dmc_profit = make_scorer(dmc_profit, greater_is_better=True)



# takes models and calculates the cv profit
# as shown by the chair
def cv_profits_for_models(models, X, y):
    for model in models:
        cv = StratifiedKFold(n_splits=10, random_state=42)
        profit = sum(cross_validate(model, X, y=y, cv=10, scoring=profit_scoring)['test_score'])
        print("Model: " + type(model).__name__)
        print("Estimated Profit: " + str(profit))
        print("")
    
    
# takes any sklearn model, X, y , num folds --> none
# outputs a plot of cross validation mean scores of the custom profit function
# for different prediction thresholds (e. g. predicting fraud=1 if probab = x)
def plot_cv_confidence_vs_profit(model, X, y,cvfolds, modelname="ModelName"): 
    thresholds = []
    profits = []          
    for threshold in np.arange(0.3,1,0.01):
        # wrap the model s. t it predicts with the threshold
        wrap =  CustomModelWithThreshold(model,threshold)
        # calc cross val mean custom score
        # you have to multiply with the number of folds s. t. you get
        # an average for the complete dataset
        profit = sum((cross_val_score(wrap, X, y, cv=cvfolds, scoring=profit_scoring)))
        thresholds.append(threshold)
        profits.append(profit)
    
    fig,ax = plt.subplots()
    ax.plot(thresholds, profits)  
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Profit")
    plt.title("Cross Validation Profits for different Prediction Thresholds ({})".format(modelname))  
    
    mx = max(profits)
    maxindex = profits.index(mx)
    thrsh = thresholds[maxindex]
    thrsh = round(thrsh,2)
    ax.annotate("Trsh:" +  str(thrsh) +" " + "Profit: " + str(mx), (thresholds[maxindex], profits[maxindex]))
    

# calculates cross-validation predictions and prints a confusion matrix
# takes a sklearn model and dataset   
# the threshold is the probab threshold for the model to predict class 1    
# (sklearn model, Xtrain, y, threshold) --> dataset with predictions/probabilities and true labels    
def cv_preds_and_confusion_matrix(model_, X, y, cvfolds, threshold=0.5):
    cvres = pd.DataFrame()
    model = CustomModelWithThreshold(model_,threshold)
    cvproba = cross_val_predict(model,X,y,cv=cvfolds, method="predict_proba")
    cvpredict = cross_val_predict(model,X,y,cv=cvfolds)
    cvres["true"] = y
    cvres['cvproba'] = cvproba[:,1]
    cvres['cvpredict'] = cvpredict
    TP = sum((cvres['true'] == 1) & (cvres['cvpredict'] == 1))
    TN = sum((cvres['true'] == 0) & (cvres['cvpredict'] == 0))
    FP = sum((cvres['true'] == 0) & (cvres['cvpredict'] == 1))
    FN = sum((cvres['true'] == 1) & (cvres['cvpredict'] == 0))
    profit = (5*TP -25*FP - 5*FN)
    print('       #### Confusion Matrix ####')
    print(' True negatives: {}'.format(TN) + '   ' + 'False Negatives: {}'.format(FN))
    print(' False positives: {}'.format(FP) + '   ' + 'True Positives: {}'.format(TP))
    print(' ')
    print('Estimated Profit: {}'.format(profit))
    return cvres


  
    