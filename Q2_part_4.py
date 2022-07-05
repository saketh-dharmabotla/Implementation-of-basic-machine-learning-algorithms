# Question 2 part 4
# Implementing gradient descent with ridge regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('A2Q2Data_train.csv', header = None) # reading the training data

D = np.array(data)                   # each row of D correspond to the data points,
                                     # where the first 100 components are the features and the last one is the label

X = np.transpose(D[:, 0:100])        # X contains data points in its columns

Y = D[:, 100]                        # Y contains all the labels
A = np.matmul(X, Y)

C = np.matmul(X, np.transpose(X))    # Covariance matrix
P = np.linalg.pinv(C)                # psuedo inverse of the covariance matrix

w_ML = np.matmul(P, A)               # least squares solution

# 80 - 20 cross validation

X_train = X[:, 0:8000]               
Y_train = Y[0:8000] 
C_train = np.matmul(X_train, np.transpose(X_train))
A_train = np.matmul(X_train, Y_train)

X_valid = X[:, 8000:10000]           
Y_valid = Y[8000:10000]
C_valid = np.matmul(X_valid, np.transpose(X_valid))

# initialization
rows, cols = X.shape

plt.title("Ridge Regression (error vs lambda)")
plt.xlabel("lambda")
plt.ylabel("error on validation set")

# for loop finds the lambda which has the least error using cross validation
for lam in range(2, 11):

    w = np.random.rand(100)
    t = 1
    grad = 2*(np.matmul(C, w) - A_train) + lam*w  # intial gradient

    while np.linalg.norm(grad) > 0.01:
    
        grad = 2*(np.matmul(C_train, w) - A_train) + lam*w # gradient
        
        if np.linalg.norm(grad) > 10:
            grad /= np.linalg.norm(grad)
        
        w = w - 1/t * (grad)
        t += 1

    # computing the error on the validation set
    Y_pred = np.matmul(np.transpose(X_valid), w)
    error = (np.linalg.norm(Y_pred - Y_valid))**2
    print(error)
    plt.plot(lam, error, 'bo')

plt.show()

# computing w_R using lambda = 7

lam = 7

Q = np.linalg.pinv(C + lam*np.identity(100))
w_R = np.matmul(Q, A)
print("\n The optimal value of lambda is 7 \n w_R for lambda = 7 is : \n", w_R)

# testing the model

test_data = pd.read_csv('A2Q2Data_test.csv', header = None) # reading the test data
E = np.array(test_data)

X_test = np.transpose(E[:, 0:100])
Y_test = E[:, 100]

# using linear regression
Y_ML = np.matmul(np.transpose(X_test), w_ML)
error_ML = (np.linalg.norm(Y_test - Y_ML))**2

# using ridge regression 
Y_R = np.matmul(np.transpose(X_test), w_R)
error_R = (np.linalg.norm(Y_test - Y_R))**2

print("\n The error on the test data when using linear regression is : ", error_ML)
print("\n The error on the test data when using ridge regression is : ", error_R)