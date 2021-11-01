#!/usr/bin/env python3

# Notation (from lecture 3): 
#     Training data has samples i = 1,...,m
#     x_i is data point (length n vector), s_i is label (+1 or -1)
#     a (length n vector) is weights, b (scalar) is bias
#     t_i are slack variables
#
# We solve the LP:
#     minimize the sum of all t_i
#     s.t. 0 <= t_i, i=1,..,m
#          1 - s_i*((a^T)x_i + b) <= t_i
#
# Or with SVM:
#    minimize the sum of all [max{0, 1 - s_i((a^T)x_i + b)} + lambda*|a|^2]
#    Question: What is lambda? Looks like it could be a hyperparameter we choose.

#######
# this may be useful for svm implementation:  https://www.cvxpy.org/examples/machine_learning/svm.html
#######

### imports
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

### helper functions:

### MyClassifier Class
class MyClassifier:
    """Main Class for the Project
    
    Attributes:

        X_train: the training data vectors
        Y_train: the training data labels
        X_test: the testing data vectors
        Y_test: the testting data labels
        
        n: how many dimensions our data input is
            e.g. the MNIST would have 784 as n, as it is a 28x28 image
        m: how many training samples we have
        a: the weight vector [n x 1]
        b: the bias vector [1]
        lambda: a hyperparameter (TODO: tune this)
    
    """
    def __init__(self, data):
        """initializes the data
        arg:
            data: a list consisting of [X_train, y_train, X_test, y_test]
        """
        np.random.seed(1) # for reproducibility
        
        self.X_train = data[0]
        self.y_train = data[1]
        self.X_test = data[2]
        self.y_test = data[3]
                      
        # initializing weights
        self.n = self.X_train.shape[1]
        self.m = self.X_train.shape[0]
        self.a = np.zeros(self.n)
        self.b = 0
        self.lambda = 1e-1

    def sample_selection(self, training_sample):
        if True: # TODO
            self.training_data.append(training_sample)
        
        return self

    def train(self, train_data, train_label):
        '''

        Args:
            train_data: N_train x M
                        M: number of features in data vectors (784) for MNIST
                        N_train: number of points used for training
                        Each row of train_data corresponds to a data point
            train_label: vector of length N_train

        Returns:
            MyClassifier object

        '''



        return self

    def f(self, input):
        '''

        Args:
            input: vector of length L
                   corresponds to the function g(y)

        Returns:
            estimated class

        '''

        if np.dot(self.a, input) + self.b > 0:
            return 1
        elif np.dot(self.a, input) + self.b <= 1:
            return -1

    def test(self, test_data):
        '''

        Args:
            test_data: m_test x n_test size matrix where
                       m_test: number of features
                       n_test: number of test data

        Returns:
            vector that contains the classification decisions

        '''
        return np.apply_along_axis(self.f, 0, test_data)


