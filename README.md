# GraphicalApproachesAndModel

Assignment
This assignment will let you explore the use of Bayesian Networks on actual data based on the problem of insurance fraud detection. You will need to mobilise what you have done on practical segments 1 to 4, as we will not be needing Dynamic Networks for this problem. 
You will work on the following dataset from Kaggle: 

https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data

You will work in teams of 3 students.

The requirements are the following: 

Create a custom sckikit-learn classifier based on the template provided. You should let users either provide an exiting bayesian network model. If a model isn't  provided, you should automatically perform structure learning in fit, before the estimation of the parameters. 
Apply the new Bayesian Network classifier to the dataset using an scikit-learn pipeline
Benchmark against other one other scikit learn classifier
Amongst the wrongfully classified claims, select 5 instances and use the structure of the learned bayesian network, to interpret the misclassification. 
Template for an sklearn classifier: 

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class TemplateClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y):

        # Check that X and y have correct shape, set n_features_in_, etc.
        X, y = validate_data(self, X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        
        # If bayesian network model not provided at init, perform structure learning

        # Estimate parameters from data

        # Return the classifier
        return self

    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = validate_data(self, X, reset=False)
       
        # Perform prediction here with predict_proba
        # Return just result (without the probability)

    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = validate_data(self, X, reset=False)

        # Implement prediction here, return classes with probabilities