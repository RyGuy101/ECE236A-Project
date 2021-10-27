#!/usr/bin/env python3

# Notation (from lecture 3): 
#     Training data has samples i = 1,...,m
#     x_i is data point (length n vector), s_i is label (+1 or -1)
#     a (length n vector) is weights, b (length n vector) is biases
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

import numpy as np
class MyClassifier:
    def __init__(self, n):
        self.training_data = []
        self.a = np.zeros(n)
        self.b = np.zeros(n)

    def sample_selection(self, training_sample):
        if True: # TODO
            self.training_data.append(training_sample)
        
        return self

    def train(self, train_data, train_label):
        return self

    def f(self, input):
        pass

    def test(self, test_data):
        pass
