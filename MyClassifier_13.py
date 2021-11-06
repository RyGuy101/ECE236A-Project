#!/usr/bin/env python3

# Notation from lecture 3: 
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

# Notation in Project Description (used in the code):
#     Training data has samples i = 1,...,N
#     y_i is data point (length M vector), s_i is label (+1 or -1)
#     W (M by L matrix) is weights, w (length L vector) is bias (L is chosen by us)
#     g(y) = (W^T)y + w
#     f(x) = f(g(y)) decides classification label
#
# Example LP for L = 1:
#     minimize the sum of all t_i
#     s.t. 0 <= t_i, i=1,..,N
#          1 - s_i*((W^T)y_i + w) <= t_i

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

        y_train: the training data vectors
        s_train: the training data labels
        y_test: the testing data vectors
        s_test: the testting data labels
        
        M: how many dimensions our data input is
            e.g. the MNIST would have 784 as n, as it is a 28x28 image
        N: how many training samples we have
        W: the weight vector [M x 1]
        w: the bias vector [1]
        lambda: a hyperparameter (TODO: tune this)
    
    """
    def __init__(self, data):
        """initializes the data
        arg:
            data: a list consisting of [y_train, s_train, y_test, s_test]
        """
        np.random.seed(1) # for reproducibility
        
        self.y_train = data[0]
        self.s_train = data[1]
        self.y_test = data[2]
        self.s_test = data[3]
                      
        # initializing weights
        self.M = self.y_train.shape[1]
        self.W = np.zeros(self.M)
        self.w = 0
        # self.lambda = 1e-1

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
        # https://www.cvxpy.org/examples/basic/linear_program.html
        N_train = train_data.shape[0]
        Y = train_data
        S = train_label
        W = cp.variable(self.M) # Assumes L = 1
        w = cp.variable(1)
        t = cp.variable(N_train)
        prob = cp.Problem(cp.Minimize(np.ones(N_train)@t), [
            np.zeros(N_train) <= t, # 0 <= t_i, i=1,..,N
            np.ones(N_train) - S*(Y@W + w*np.ones(N_train)) <= t # 1 - s_i*((W^T)y_i + w) <= t_i
        ])
        prob.solve()
        print("\nThe optimal value is", prob.value)
        print("A solution W, w is")
        print("W = {}".format(W))
        print("w = {}".format(w))

        self.W = W
        self.w = w
        return self

    def f(self, input):
        '''

        Args:
            input: vector of length L
                   corresponds to the function f(x) = f(g(y))

        Returns:
            estimated class

        '''

        if abs(input) < 1:
            print("Unsure about classification. Value is {}".format(input))

        # decision function in Project Description
        if input > 0:
            return 1
        elif input <= 0:
            return -1

    def test(self, test_data):
        '''

        Args:
            test_data: N_test x M size matrix where
                       M: number of features
                       N_test: number of test data

        Returns:
            vector that contains the classification decisions

        '''
        N_test = test_data.shape[0]
        return np.apply_along_axis(self.f, 0, test_data@self.W + self.w*np.ones(N_test))
