#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import random
from sklearn.linear_model import LogisticRegression as LR

# see https://sklearn-template.readthedocs.io/en/latest/user_guide.html
# for documentation about how to write a custom model


# this model takes any classifier model and a threshold as input
# the predict of this model is the predict of the input model but the
# prediction threshold is not 0.5 instead it is the custom threshold
class CustomModelWithThreshold(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold):
        #sklearn model object
        self.model = model
        self.threshold = threshold
          
    def fit(self,X,y):
        self.classes_ = np.unique(y)
        self.model.fit(X,y)
        return self

    def predict(self,X):
        preds = self.model.predict_proba(X)
        preds[preds>self.threshold] = 1
        preds[preds<= self.threshold] = 0
        return preds[:,1]

    def predict_proba(self,X):
        return self.model.predict_proba(X)
    
    
# perceptron
# learns with the pocket variant of the perceptron learning algorithm 
# the learning process here is adapted to the DMC cost matrix    
class PerceptronLearner(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs): 
        self.epochs = epochs
        self.coef_ = None
        
    def predict(self, X0):
         # add intercept (bias)
        X = np.zeros((X0.shape[0], X0.shape[1]+1))
        X[:,0] = 1
        X[:,1:] = X0
        
        # calc scores
        sc = np.dot(X,self.coef_)
        sc[sc>=0]=1
        sc[sc<0]=0
        return sc
    
    

    def predict_proba(self,X0):
        # there are actually no probabs so use the predicions
        pr = np.zeros((X0.shape[0],2))
        pr[:,1] = self.predict(X0)
        pr[:,0] = 1 - pr[:,1]
        return pr
    
    
    # implementation of Pocket algorithm for Perceptron learning
    def fit(self,X0,y):
        
        self.classes_ = np.unique(y)
        
        ### adapted Pocket algorithm
      
        # add intercept (bias)
        X = np.zeros((X0.shape[0], X0.shape[1]+1))
        X[:,0] = 1
        X[:,1:] = X0
        N, D = X.shape
         # change here to initialize weights in a different way
        w = np.zeros(D)
       
        
        y_true = y.copy()
        y_true[y_true==0] = -1
        epochcounter = 0
            
        profit_matrix = {(-1,-1): 0, (-1,1): -5, (1,-1): -25, (1,1): 5}
        profits = []
        weight_sequence = []
        
        # convert to list to prevent key error in cross val
        y_true = list(y_true)
        
        # note: we keep current profit over epochs
        # and only reset profit when we hit an error
        current_profit = 0
        print('Start pocket algorithm with {} epochs'.format(self.epochs))
        while(epochcounter<self.epochs):
            if epochcounter % 50 == 0:
                print('Epoch: {}'.format(epochcounter))
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
                # different error types are not weighted (yet?)
                if (y_hat_i != y_true_i):
                     profits.append(current_profit)
                     current_profit = 0
                     weight_sequence.append(w)
                     w = w + y_true_i * X[idx]    
            
        # set best weights
        self.coef_ = weight_sequence[profits.index(max(profits))]
        return self
        
        
      
     
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    