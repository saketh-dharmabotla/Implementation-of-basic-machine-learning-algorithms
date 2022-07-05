# Question 2 part 2 
# implementation of gradient descent to solve the least squares problem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('A2Q2Data_train.csv', header = None) # reading the data

D = np.array(data)                   # each row of D correspond to the data points,
                                     # where the first 100 components are the features and the last one is the label

X = D[:, 0:100]                      # X contains the features of each data point in its rows
X = np.transpose(X)                  # X is updated to its transpose

Y = D[:, 100]                        # Y contains all the labels
A = np.matmul(X, Y)

C = np.matmul(X, np.transpose(X))    # Covariance matrix
P = np.linalg.pinv(C)                # psuedo inverse of the covariance matrix

w_ML = np.matmul(P, np.matmul(X, Y)) # least squares solution

# gradient descent
rows, cols = X.shape

# initialization
w = np.random.rand(100)
t = 1
grad = 2*(np.matmul(C, w) - A)  # intial gradient

plt.title("||w - w_ML|| vs iteration number")
plt.xlabel("Iteration number")
plt.ylabel("||w - w_ML||")

while np.linalg.norm(grad) > 0.01:
    
    error = np.linalg.norm(w - w_ML)
    plt.plot(t, error, 'bo', markersize = 2)
    
    grad = 2*(np.matmul(C, w) - A)  # gradient
    
    if np.linalg.norm(grad) > 10:
        grad /= np.linalg.norm(grad)
    
    w = w - 1/t * (grad)
    t += 1
    
error = np.linalg.norm(w - w_ML)
plt.plot(t, error, 'bo', markersize = 2)

plt.show()

print("The L2 norm of the error at the end of gradient descent is :")
print(np.linalg.norm(w - w_ML))