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
    
    #A lot of these features seem to be irrelevant and even reduce the score
    #Maybe reduce the default case to only scannedLineItemsTotal?
    def __init__(self, featurelist = ['scannedLineItemsTotal',
                                     'valuePerLineItem',
                                     'quantityModificationsPerLineItem',
                                     'totalScanTimeInSeconds*lineItemVoids',
                                     'totalScanTimeInSeconds*scansWithoutRegistration',
                                     'totalScanTimeInSeconds*scannedLineItemsTotal',
                                     'lineItemVoids*scansWithoutRegistration',
                                     'totalScanTimeInSeconds/trustLevel',
                                     'lineItemVoids/trustLevel',
                                     'scansWithoutRegistration/trustLevel',
                                     'scannedLineItemsTotal/trustLevel',
                                     'trustLevel_Log',
                                     'grandTotal_Log',
                                     'quantityModifications_Square',
                                     'scannedLineItemsTotal_Square']):
    
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
        
        #interesting features from feature_engineering notebook
        #polynomial features
        
        if 'totalScanTimeInSeconds*lineItemVoids' in self.featurelist:
            X['totalScanTimeInSeconds*lineItemVoids'] = X['totalScanTimeInSeconds'] * X['lineItemVoids']
        
        if 'totalScanTimeInSeconds*scansWithoutRegistration' in self.featurelist:
            X['totalScanTimeInSeconds*scansWithoutRegistration'] = X['totalScanTimeInSeconds'] * X['scansWithoutRegistration']
        
        if 'totalScanTimeInSeconds*scannedLineItemsTotal' in self.featurelist:
            X['totalScanTimeInSeconds*scannedLineItemsTotal'] = X['totalScanTimeInSeconds'] * X['scannedLineItemsTotal']
        
        if 'lineItemVoids*scansWithoutRegistration' in self.featurelist:
            X['lineItemVoids*scansWithoutRegistration'] = X['lineItemVoids'] * X['scansWithoutRegistration']
        
        
        #division features
        #!!! Be carefull with division by 0 !!!
        #right now only divison by trustLevel, which is never zero
        if 'totalScanTimeInSeconds/trustLevel' in self.featurelist:
            X['totalScanTimeInSeconds/trustLevel'] = X['totalScanTimeInSeconds'] / X['trustLevel']
        
        if 'lineItemVoids/trustLevel' in self.featurelist:
            X['lineItemVoids/trustLevel'] = X['lineItemVoids'] / X['trustLevel']
        
        if 'scansWithoutRegistration/trustLevel' in self.featurelist:
            X['scansWithoutRegistration/trustLevel'] = X['scansWithoutRegistration'] / X['trustLevel']
        
        if 'scannedLineItemsTotal/trustLevel' in self.featurelist:
            X['scannedLineItemsTotal/trustLevel'] = X['scannedLineItemsTotal'] / X['trustLevel']
        
        
        #Log features
        if 'trustLevel_Log' in self.featurelist:
            X['trustLevel_Log'] = np.log(X['trustLevel'])
        
        if 'grandTotal_Log' in self.featurelist:
            X['grandTotal_Log'] = np.log(X['grandTotal'])
        
        #Square features
        if 'quantityModifications_Square' in self.featurelist:
            X['quantityModifications_Square'] = np.square(X['quantityModifications'])
        
        if 'scannedLineItemsTotal_Square' in self.featurelist:
            X['scannedLineItemsTotal_Square'] = np.square(X['scannedLineItemsTotal'])
        
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
    
    
    
    
    
    