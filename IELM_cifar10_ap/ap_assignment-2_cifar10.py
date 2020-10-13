
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 4 2019

@author: Ankit Patel
"""
from keras.datasets import cifar10
from keras.utils import to_categorical
from ielm_cifar10 import I_ELM
import numpy as np

def main():
    # ===============================
    # Define Constants
    # ===============================
    no_classes = 10
    Lmax = 100
    error = 0.1
    loss_function = "mean_squared_error"  #It can be mean_absolute_error also
    activation_function = "sigmoid"
    
    # ===============================
    # Load dataset using Keras module
    # ===============================
    (X_Train, Y_Train), (X_Test, Y_Test) = cifar10.load_data()

    # ===============================
    # Preprocess dataset by normalizing and converting outputs to categorical values
    # ===============================
    X_Train = X_Train.astype(np.float32) / 255.
    X_Train = X_Train.reshape(-1, 3*(32**2))
    X_Test = X_Test.astype(np.float32) / 255.
    X_Test = X_Test.reshape(-1, 3*(32**2))
    Y_Train = to_categorical(Y_Train, no_classes).astype(np.float32)
    Y_Test = to_categorical(Y_Test, no_classes).astype(np.float32)

    # ===============================
    # Instantiate ELM object
    # ===============================
    model = I_ELM(
        no_input_nodes=3*(32**2),
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
    training_loss, training_acc = model.evaluate(X_Train, Y_Train, metrics=['loss', 'accuracy'])
    print('Training Loss in mean square error: %f' % training_loss) # loss value
    print('Training Accuracy: %f' % training_acc)# accuracy
    print('Total Time require for Training %f Seconds'% (final-i))
    

    # ===============================
    # Test
    # ===============================
    i = time.time()
    test_loss, test_acc = model.evaluate(X_Test, Y_Test, metrics=['loss', 'accuracy'])
    final = time.time()
    print('Testing Loss in mean square error: %f' % test_loss)
    print('Testing Accuracy: %f' % test_acc)
    print('Total Time require for Testing one image is %f Seconds'% ((final-i)/X_Test.shape[0]))
        
if __name__ == '__main__':
    main()
