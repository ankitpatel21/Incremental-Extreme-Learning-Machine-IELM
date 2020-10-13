
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 4 2019

@author: Ankit
"""
import numpy as np

class I_ELM():
    """ Constructor to initialize node"""
    def __init__(self, no_input_nodes, max_no_hidden_nodes, no_output_nodes,
        activation_function='sigmoid', loss_function='mean_squared_error'):

        #self.name = name
        self.no_input_nodes = no_input_nodes
        self.no_hidden_nodes = 1
        self.no_output_nodes = no_output_nodes

        # initialize weights between  hidden layer and Output Layer
        self.beta = np.random.uniform(-1.,1.,size=(self.no_hidden_nodes, self.no_output_nodes))
        # initialize weights between Input Layer and hidden layer
        self.alpha = np.random.uniform(-1.,1.,size=(self.no_input_nodes, self.no_hidden_nodes))
        #Initialize Biases
        self.bias = np.zeros(shape=(self.no_hidden_nodes,))
        # set an activation function
        self.activation_function = activation_function
        # set a loss function
        self.loss_function = loss_function
    
    def mean_squared_error(self,Y_True, Y_Pred):
        return 0.5 * np.mean((Y_True - Y_Pred)**2)

    def mean_absolute_error(self, Y_True, Y_Pred):
        return np.mean(np.abs(Y_True - Y_Pred))
    
    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def predict(self, X):
        return list(self(X))
    
    def __call__(self, X):
        h = self.sigmoid(X.dot(self.alpha) + self.bias)
        return h.dot(self.beta)

    def evaluate(self, X, Y_true, metrics=['loss']):
        Y_pred = self.predict(X)
        Y_true = Y_true
        Y_pred_argmax = np.argmax(Y_pred, axis=-1)
        Y_true_argmax = np.argmax(Y_true, axis=-1)
        ret = []
        for m in metrics:
            if m == 'loss':
                loss = self.mean_squared_error(Y_true, Y_pred)
                ret.append(loss)
            elif m == 'accuracy':
                acc = np.sum(Y_pred_argmax == Y_true_argmax) / len(Y_true)
                ret.append(acc)
            else:
                raise ValueError('an unknown evaluation indicator \'%s\'.' % m)
        if len(ret) == 1:
            ret = ret[0]
        elif len(ret) == 0:
            ret = None
        return ret

    def fit(self, X, Y_true,Lmax,error):
        self.beta = np.random.uniform(-1.,1.,size=(1, self.no_output_nodes))
        self.alpha = np.random.uniform(-1.,1.,size=(self.no_input_nodes, 1))
        print(self.beta.shape,self.alpha.shape)
        H = self.sigmoid(X.dot(self.alpha))
        # compute a pseudoinverse of H
        H_pinv = np.linalg.pinv(H)
        # update beta
        self.beta = H_pinv.dot(Y_true)

        
        for i in range(2,Lmax):
            beta_random = np.random.uniform(-1.,1.,size=(1, self.no_output_nodes))
            alpha_random = np.random.uniform(-1.,1.,size=(self.no_input_nodes, 1))
            self.alpha=np.hstack([self.alpha,alpha_random])
            print(self.beta.shape,beta_random.shape)
            self.beta = np.vstack([self.beta,beta_random])
            H = self.sigmoid(X.dot(self.alpha))
            # compute a pseudoinverse of H
            H_pinv = np.linalg.pinv(H)
            # update beta
            self.beta = H_pinv.dot(Y_true)

            

    


