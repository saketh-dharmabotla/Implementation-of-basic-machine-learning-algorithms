# Question 1 part 2
# implementation of EM algorithm for a Gaussian mixture

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math 

# reading the data
data = pd.read_csv('A2Q1.csv', header = None) 
X = np.transpose(np.array(data))
print(X.shape)

mu = np.zeros(4)
var = np.zeros(4)
pi = np.zeros(4)
muPrev = np.random.rand(4)*10
varPrev = np.random.rand(4)*10
piPrev = np.random.rand(4)*10
piPrev = piPrev/np.sum(piPrev)
lam = np.zeros((4, 1000))
denominator = np.zeros(1000)

epsilon = 0.01                  # tolerance chosen is 0.01
t = 0
difference = 100

mLL = 0                         # stores the value of the modified log likelihood at the current iteration
avgMLL_list = []                # stores the average value of the modified log likelihood over 100 random initializations
count = np.zeros(100)           # stores the total number of iterations + 1 in each random initialization
lastValue = np.zeros(100)       # stores the last value of modified log likelihood in each random initialization 


while difference > epsilon:
    mu = np.zeros(4)
    var = np.zeros(4)
    pi = np.zeros(4)
    denominator = np.zeros(1000)

    # computing lambda
    for i in range(0, 1000):
        for l in range(0, 4):
            denominator[i] += piPrev[l]*math.exp(-((X[:, i] - muPrev[l])**2)/(2*varPrev[l]))/(2*np.pi*varPrev[l])**0.5

    for l in range(0, 4):
        for i in range(0, 1000):
            lam[l][i] = piPrev[l]*(math.exp(-((X[:, i] - muPrev[l])**2)/(2*varPrev[l]))/(2*np.pi*varPrev[l])**0.5)/denominator[i]           

    # computing mu
    for l in range(0, 4):
        numeratorMu = 0
        denominatorMu = 0
        
        for i in range(0, 1000):
            numeratorMu += lam[l][i]*X[:, i] 
            denominatorMu += lam[l][i]
        
        mu[l] = numeratorMu/denominatorMu    

    # computing variances
    for l in range(0, 4):
        numeratorVar = 0
        denominatorVar = 0

        for i in range(0, 1000):
            numeratorVar += lam[l][i]*((X[:, i] - muPrev[l])**2)
            denominatorVar += lam[l][i]

        var[l] = numeratorVar/denominatorVar 

    # computing pi
    for l in range(0, 4):
        numeratorPi = 0
        
        for i in range(0, 1000):
            numeratorPi += lam[l][i]
        
        pi[l] = numeratorPi/1000

    # computing log likelihood
    mLL = 0
    for i in range(0, 1000):
        sum = 0

        for l in range(0, 4):
            sum += pi[l]*math.exp(-((X[:, i] - mu[l])**2)/(2*var[l]))/((2*np.pi*var[l])**0.5)

        mLL += math.log(sum)

    difference = (np.linalg.norm(mu - muPrev)**2 + np.linalg.norm(var - varPrev)**2 + np.linalg.norm(pi - piPrev)**2)**(0.5)
    t += 1

    muPrev = mu
    varPrev = var
    piPrev = pi


plt.title("Expectation Maximization Algorithm (Exponential mixture)")       
plt.xlabel("iteration number")
plt.ylabel("log likelihood")
plt.show()