#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np

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
    