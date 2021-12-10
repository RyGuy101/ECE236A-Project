import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random
import requests, gzip, os, hashlib

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
    
    """
    def __init__(self, M):
        """initializes the data
        arg:
            M: number of features
        """
                      
        # initializing weights
        self.M = M
        self.W = np.zeros(self.M)
        self.w = 0

        # initialize empty training set
        self.y_train = np.zeros((0, self.M))
        self.s_train = np.zeros(0, dtype=np.int8)
        self.linearly_separable = True # assume linearly separable data to start

    def sample_selection(self, training_sample, training_label): # training_label added as an input, as mentioned on Campuswire
        '''

        Args:
            training_sample: vector of length M
                        M: number of features in data vectors (784) for MNIST
            training_label: class label (1 or -1)

        Returns:
            MyClassifier object

        '''
        g_y = training_sample@self.W + self.w
        penalty = 1 - (training_label * g_y)

        # penalty <= 0     -> classified perfectly correctly
        # 0 < penalty < 1  -> classified correctly but within margin
        # penalty >= 1     -> classified incorrectly

        # Select points that are classified incorrectly, or classified correctly and some distance within the margin (need to include some correct classifications to balance out noisy data (e.g. synthetic dataset))
        if penalty >= 0.3125:
            self.y_train = np.append(self.y_train, [training_sample], axis=0)
            self.s_train = np.append(self.s_train, [training_label], axis=0)

        return self


    def LP(self,training_set,training_labels):
        #First, generate the best decision hyperplane using all data samples
        N_train = training_set.shape[0]
        norm_dist = training_set.shape[1]
        #Set up the LP
        #######################################################################
        # CHANGE THIS NUM FOR HOW MANY N SAMPLES YOU WANT LP TO USE
        no_samp = 40
        #######################################################################
        y_train = training_set
        s_train = training_labels

        #Get class means
        mn_neg1 = np.mean(y_train[s_train==-1],axis=0)
        mn_1 = np.mean(y_train[s_train==1],axis=0)
        center = (mn_neg1+mn_1)/2

        #approx. margin estimates
        margin_dir = mn_1-mn_neg1
        margin_dir = margin_dir/np.linalg.norm(margin_dir)

        #sample wise margin estimates
        y_res = y_train-np.expand_dims(center,axis=0)


        g_y = (y_res@margin_dir)*s_train
        g_y_2 = np.diag(g_y)
        #Representative sampling- so that chosen samples are representative of the original distributions
        n = cp.Variable(shape=(N_train))
        v = cp.Variable(1)
        v2 = cp.Variable(1)

        # W_new = margin_dir
        N_train_1 = len(s_train[s_train==1])
        N_train_neg1 = len(s_train[s_train==-1])
        

        #distance constraints
        dist_neg1 = distance.cdist(y_train[s_train==-1],y_train[s_train==-1])/norm_dist
        dist_neg1[dist_neg1==0] = np.amax(dist_neg1,axis=1)
        dist_1 = distance.cdist(y_train[s_train==1],y_train[s_train==1])/norm_dist
        dist_1[dist_1==0] = np.amax(dist_1,axis=1)
        dist_alt_neg1 = distance.cdist(y_train[s_train==-1],y_train[s_train==1])/norm_dist
        dist_alt_1 = distance.cdist(y_train[s_train==1],y_train[s_train==-1])/norm_dist

        prob = cp.Problem(cp.Minimize(v2-v),
            [np.ones(N_train_1)@n[s_train==1] <= int(no_samp/2),
            np.ones(N_train_1)@n[s_train==1] >= int(no_samp/2),
            np.ones(N_train_neg1)@n[s_train==-1] <= int(no_samp/2),
            np.ones(N_train_neg1)@n[s_train==-1] >= int(no_samp/2),
            g_y_2@(n) >= 0.0*np.ones(N_train),
            np.amin(dist_neg1,axis=1)@(n[s_train==-1]) >= v,
            np.amin(dist_1,axis=1)@(n[s_train==1]) >= v,
            np.mean(dist_alt_neg1,axis=1)@n[s_train==-1] <= v2,
            np.mean(dist_alt_1,axis=1)@n[s_train==1] <= v2,
            n>=np.zeros(N_train),
            n<=np.ones(N_train)
            ])
        prob.solve()

        mask = n.value > 0.5
        y_new = y_train[mask]
        s_new = s_train[mask]
        print('No. of chosen samples = {}'.format(sum(mask)))
        return y_new, s_new

    def ILP(self,training_set,training_labels):
        #First, generate the best decision hyperplane using all data samples
        N_train = training_set.shape[0]
        norm_dist = training_set.shape[1]
        #######################################################################
        # CHANGE THIS NUM FOR HOW MANY N SAMPLES YOU WANT LP TO USE
        no_samp = 40
        #######################################################################
        y_train = training_set
        s_train = training_labels

        #Get class means
        mn_neg1 = np.mean(y_train[s_train==-1],axis=0)
        mn_1 = np.mean(y_train[s_train==1],axis=0)
        center = (mn_neg1+mn_1)/2

        #approx. margin estimates
        margin_dir = mn_1-mn_neg1
        margin_dir = margin_dir/np.linalg.norm(margin_dir)

        #sample wise margin estimates
        y_res = y_train-np.expand_dims(center,axis=0)

        g_y = (y_res@margin_dir)*s_train
        g_y_2 = np.diag(g_y)
        #Representative sampling- so that chosen samples are representative of the original distributions
        n = cp.Variable(shape=(N_train),integer=True) #)#
        v = cp.Variable(1, integer=True)
        v2 = cp.Variable(1, integer=True)

        # W_new = margin_dir
        N_train_1 = len(s_train[s_train==1])
        N_train_neg1 = len(s_train[s_train==-1])

        #distance constraints
        dist_neg1 = distance.cdist(y_train[s_train==-1],y_train[s_train==-1])/norm_dist
        dist_neg1[dist_neg1==0] = np.amax(dist_neg1,axis=1)
        dist_1 = distance.cdist(y_train[s_train==1],y_train[s_train==1])/norm_dist
        dist_1[dist_1==0] = np.amax(dist_1,axis=1)
        dist_alt_neg1 = distance.cdist(y_train[s_train==-1],y_train[s_train==1])/norm_dist
        dist_alt_1 = distance.cdist(y_train[s_train==1],y_train[s_train==-1])/norm_dist

        prob = cp.Problem(cp.Minimize(v2-v),
            [np.ones(N_train_1)@n[s_train==1] <= int(no_samp/2),
            np.ones(N_train_1)@n[s_train==1] >= int(no_samp/2),
            np.ones(N_train_neg1)@n[s_train==-1] <= int(no_samp/2),
            np.ones(N_train_neg1)@n[s_train==-1] >= int(no_samp/2),
            g_y_2@(n) >= 0.0*np.ones(N_train),
            np.amin(dist_neg1,axis=1)@(n[s_train==-1]) >= v,
            np.amin(dist_1,axis=1)@(n[s_train==1]) >= v, 
            np.mean(dist_alt_neg1,axis=1)@n[s_train==-1] <= v2,
            np.mean(dist_alt_1,axis=1)@n[s_train==1] <= v2,
            n>=np.zeros(N_train),
            n<=np.ones(N_train)
            ])
        prob.solve()

        mask = n.value > 0.5
        y_new = y_train[mask]
        s_new = s_train[mask]
        print('No. of chosen samples = {}'.format(sum(mask)))
        return y_new, s_new


    def train(self, train_data=None, train_label=None):
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
        if train_data is None:
            train_data = self.y_train
        if train_label is None:
            train_label = self.s_train

        # https://www.cvxpy.org/examples/basic/linear_program.html
        N_train = train_data.shape[0]
        Y = train_data
        S = train_label

        if not np.any(S == 1) or not np.any(S == -1):
            return # Do nothing. Need both classes to train effectively

        W = cp.Variable(self.M) # Assumes L = 1
        w = cp.Variable(1)

        if self.linearly_separable:
            s = cp.Variable(self.M)
            r = cp.Variable(1)
            prob = cp.Problem(cp.Minimize((1+1/np.sqrt(self.M))*np.ones(self.M)@s + (1+np.sqrt(self.M))*r), [ # approximation of euclidean norm
                W <= s,
                W >= -s,
                W <= r,
                W >= -r,
                1 - (Y[S == 1]@W + w) <= 0,
                1 + (Y[S == -1]@W + w) <= 0
            ])
            prob.solve()
            if prob.status == "infeasible":
                self.linearly_separable = False
                # print("Data is not linearly separable")
        
        if not self.linearly_separable:
            t = cp.Variable(N_train)
            prob = cp.Problem(cp.Minimize(np.ones(N_train)@t), [
                np.zeros(N_train) <= t, # 0 <= t_i, i=1,..,N
                1 - (Y[S == 1]@W + w) <= t[S == 1], # 1 - s_i*((W^T)y_i + w) <= t_i
                1 + (Y[S == -1]@W + w) <= t[S == -1]
            ])
            prob.solve()

        self.W = W.value
        self.w = w.value
        return self

    def f(self, input):
        '''
        Args:
            input: vector of length L
                   corresponds to the function f(x) = f(g(y))
        Returns:
            estimated class
        '''
        if input > 0:
            return 1
        elif input < 0:
            return -1
        else:
            return np.random.choice([1, -1])

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

        return np.vectorize(self.f)(test_data@self.W + self.w*np.ones(N_test))
