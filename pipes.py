#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# sklearn pipelines


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
import numpy as np





# custom attribute adder
class CustomAttributeAdder(BaseEstimator, TransformerMixin):
    
    
    def __init__(self, featurelist = ['scannedLineItemsTotal',
                                       'valuePerLineItem',
                                       'quantityModificationsPerLineItem']):
    
        # if you use "_featurelist" sklearn will not set this in gridSearch instead it sets keys of get_params which is "featurelist" 
        self.featurelist = featurelist
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        X['scannedLineItemsTotal'] = X['scannedLineItemsPerSecond'] * X['totalScanTimeInSeconds']
            
        if "valuePerLineItem" in self.featurelist:
            X['valuePerLineItem'] = X['grandTotal'] / X['scannedLineItemsTotal']
            
        if "quantityModificationsPerLineItem" in self.featurelist:
            X['quantityModificationsPerLineItem'] = X['quantityModifications'] / X['scannedLineItemsTotal']
            
        return X
    

# custom attribute adder
class RandomAttributeAdder(TransformerMixin):
    
    """This class is still empty and needs to be filled!"""
    
    def __init__(self,):
        pass
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):    
        return X


# Implement data transformation here
class Transformer(TransformerMixin):
    
    """This class is still empty and needs to be filled!"""
    
    def __init__(self,):
        pass
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):    
        return X    
    
    
# scaling class
class Scaling(TransformerMixin):
    
    _strategy = "Standard"
    _scaler = None
    
    def __init__(self, strategy):
        self._strategy = strategy
    
    def fit(self, X, y=None):
        if self._strategy == "Standard":
            self._scaler = StandardScaler()
            return self._scaler.fit(X)
        elif self._strategy == "MinMax":
            self._scaler = MinMaxScaler()
            return self._scaler.fit(X)
        elif self._strategy == "None":
            return self
    
    def transform(self, X):
        
        if self._strategy == "None":
            return X
        else:
            return self._scaler.transform(X)
      
        
# Enables adding arbitrary sklearn models as parameter to a pipeline        
class ClfSwitcher(BaseEstimator):

    def __init__(
        self, 
        estimator = SGDClassifier()
    ):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 

        self.estimator = estimator


    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self


    def predict(self, X, y=None):
        return self.estimator.predict(X)


    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


    def score(self, X, y):
        return self.estimator.score(X, y)    
    
    
    
    
    
    