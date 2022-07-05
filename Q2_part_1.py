# Question 2 part 1
# implementing the analytic solution to the regression problem

import numpy as np
import matplotlib as plt
import pandas as pd

data = pd.read_csv('A2Q2Data_train.csv', header = None) # reading the data

D = np.array(data)                   # each row of D correspond to the data points,
                                     # where the first 100 components are the features and the last one is the label

X = D[:, 0:100]                      # X contains the features of each data point in its rows
X = np.transpose(X)                  # X is updated to its transpose

Y = D[:, 100]                        # Y contains all the labels
A = np.matmul(X, Y)

C = np.matmul(X, np.transpose(X))    # Covariance matrix
P = np.linalg.pinv(C)                # psuedo inverse of the covariance matrix

w_ML = np.matmul(P, A) # least squares solution

print(w_ML)                          # printing the solution