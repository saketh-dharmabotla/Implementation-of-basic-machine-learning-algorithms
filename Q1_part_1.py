# Question 1 part 1
# implementation of EM algorithm for an Exponential mixture

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math 

# reading the data
data = pd.read_csv('A2Q1.csv', header = None) 
X = np.transpose(np.array(data))
print(X.shape)

beta = np.zeros(4)
pi = np.zeros(4)
betaPrev = np.random.rand(4)*10
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

for k in range(0, 100):

    betaPrev = np.random.rand(4)*10
    piPrev = np.random.rand(4)*10
    piPrev = piPrev/np.sum(piPrev)
    lam = np.zeros((4, 1000))
    denominator = np.zeros(1000)
    
    t = 0
    difference = 100
    
    while difference > epsilon:
        beta = np.zeros(4)
        pi = np.zeros(4)
        denominator = np.zeros(1000)

        # computing lambda
        for i in range(0, 1000):
            for l in range(0, 4):
                denominator[i] += math.exp(-X[:, i]/betaPrev[l])*piPrev[l]/betaPrev[l] 

        for l in range(0, 4):
            for i in range(0, 1000):
                lam[l][i] = piPrev[l]*(math.exp(-X[:, i]/betaPrev[l])/betaPrev[l])/denominator[i]           

        # computing beta
        for l in range(0, 4):
            numeratorBeta = 0
            denominatorBeta = 0
            
            for i in range(0, 1000):
                numeratorBeta += lam[l][i]*X[:, i] 
                denominatorBeta += lam[l][i]
            
            beta[l] = numeratorBeta/denominatorBeta    

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
                sum += pi[l]*(math.exp(-X[:, i]/beta[l])/beta[l])

            mLL += math.log(sum)

        if(len(avgMLL_list) == 0):
            avgMLL_list.append(mLL)
        elif(t < len(avgMLL_list)):
            avgMLL_list[t] += mLL
        else:
            avgMLL_list.append(mLL)

        difference = (np.linalg.norm(beta - betaPrev)**2 + np.linalg.norm(pi - piPrev)**2)**(0.5)
        t += 1

        betaPrev = beta
        piPrev = pi


    count[k] = t
    lastValue[k] = mLL


for j in range(0, 100):
    for k in range(int(count[j]), len(avgMLL_list)):
        avgMLL_list[k] += lastValue[j]

avgMLL_list = np.array(avgMLL_list)/100 

t_max = np.amax(count)

tValues = np.arange(t_max)

plt.plot(tValues, avgMLL_list, 'bo', markersize = 2)
plt.title("Expectation Maximization Algorithm (Exponential mixture)")       
plt.xlabel("iteration number")
plt.ylabel("log likelihood")
plt.show()