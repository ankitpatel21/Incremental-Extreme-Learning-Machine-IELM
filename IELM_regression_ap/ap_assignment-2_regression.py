
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 4  2019

@author: Ankit Patel
"""

from ielm_regression import I_ELM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # ===============================
    # Define Constants
    # ===============================
    no_classes = 1
    Lmax = 100
    error = 0.1
    loss_function = "mean_squared_error"  #It can be mean_absolute_error also
    activation_function = "sigmoid"
    
    # ===============================
    # Load dataset using Keras module
    # ===============================
    dataset = pd.read_csv('Salary_Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values

    # Splitting the dataset into the Training set and Test set

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # ===============================
    # Preprocess dataset by normalizing and converting outputs to categorical values
    # ===============================
    X_Train = X_Train.astype(np.float32) 
    X_Train = X_Train.reshape(-1, 1)
    X_Test = X_Test.astype(np.float32)
    X_Test = X_Test.reshape(-1, 1)
    Y_Train = Y_Train.astype(np.float32)
    Y_Test = Y_Test.astype(np.float32)

    # ===============================
    # Instantiate ELM object
    # ===============================
    model = I_ELM(
        no_input_nodes=1,
        max_no_hidden_nodes=Lmax,
        no_output_nodes=no_classes,
        loss_function=loss_function,
        activation_function=activation_function,
        
    )

    # ===============================
    # Training
    # ===============================
    import time
    i = time.time()
    
    
    model.fit(X_Train, Y_Train,Lmax,error)
    final = time.time()
    training_loss = model.evaluate(X_Train, Y_Train, metrics=['loss'])
    print('Training Loss in mean square error: %f' % training_loss) # loss value
    
    print('Total Time require for Training %f Seconds'% (final-i))
    

    # ===============================
    # Test
    # ===============================
    i = time.time()
    test_loss = model.evaluate(X_Test, Y_Test, metrics=['loss'])
    final = time.time()
    print('Testing Loss in mean square error: %f' % test_loss)
    
    print('Total Time require for Testing one image is %f Seconds'% ((final-i)/X_Test.shape[0]))
        
if __name__ == '__main__':
    main()


