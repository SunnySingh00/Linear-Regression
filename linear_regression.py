"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd
###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    
    err = np.sum(np.absolute(X.dot(w) - y))/len(y)
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  # TODO 2: Fill in your code here #
  ##################################################### 
  w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    I = np.identity(len(X[0]))
    lambdas = 0.1+min(abs(np.linalg.eigvals(X.T.dot(X))))
    w = np.linalg.inv(X.T.dot(X)+lambdas*I).dot(X.T).dot(y)
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################

    I = np.identity(len(X[0]))
    w = np.linalg.inv(X.T.dot(X)+lambd*I).dot(X.T).dot(y)
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################
    min_mae = float('inf')
    bestlambda = None
    lambds = [10e-19,10e-18,10e-17,10e-16,10e-15,10e-14,10e-13,10e-12,10e-11,10e-10,10e-9,10e-8,10e-7,10e-6,10e-5,10e-4,10e-3,10e-2,10e-1,10e1,10e2,10e3,10e4,10e5,10e6,10e7,10e8,10e9,10e10,10e11,10e12,10e13,10e14,10e15,10e16,10e17,10e18,10e19]
    for lambd in lambds:
      w = regularized_linear_regression(Xtrain, ytrain, lambd)
      mae = mean_absolute_error(w, Xval, yval)
      if mae < min_mae:
        print(mae,lambd)
        min_mae = mae
        bestlambda = lambd     
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################
    final_array = X
    for i in range(2,power+1):
        for j in range(0,len(X[0])):
            column = np.array(X[:,j]).reshape(len(X),1)
            column = np.power(column,i)
            final_array = np.hstack((final_array, column))
    return final_array


