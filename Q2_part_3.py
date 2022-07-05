# Question 2 part 3
# Implementing Stochastic Gradient Descent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('A2Q2Data_train.csv', header = None) # reading the data

D = np.array(data)                   # each row of D correspond to the data points,
                                     # where the first 100 components are the features and the last one is the label

X = D[:, 0:100]                      # X contains the features of each data point in its rows
X = np.transpose(X)                  # X is updated to its transpose

Y = D[:, 100]                        # Y contains all the labels

C = np.matmul(X, np.transpose(X))    # Covariance matrix
P = np.linalg.pinv(C)                # psuedo inverse of the covariance matrix

w_ML = np.matmul(P, np.matmul(X, Y)) # least squares solution

# initialization
rows, cols = X.shape
w_0 = np.random.rand(100)
w_1 = np.empty(rows)
w_T = np.zeros(rows)
t = 0
T = 1000

plt.title("||w - w_ML|| vs iteration number")
plt.xlabel("Iteration number")
plt.ylabel("||w - w_ML||")

for t in range(1, T + 1):
    
    z = np.random.choice(cols, 100, replace = False)
    batchX = X[:, z]
    batchY = Y[z]
    batchC = np.matmul(batchX, np.transpose(batchX))

    error = np.linalg.norm(w_0 - w_ML)
    plt.plot(t, error, 'bo', markersize = 2)

    t += 1

    grad = 2*(np.matmul(batchC, w_0) - np.matmul(batchX, batchY))  # gradient
    
    if np.linalg.norm(grad) > 10:
        grad *= 10/np.linalg.norm(grad)

    w_t = w_0 - 1/t * grad

    w_0 = w_t
    w_T += w_0

error = np.linalg.norm(w_0 - w_ML)
plt.plot(t, error, 'bo', markersize = 2)

plt.show()

print(w_T/T)