# EM algorithm for the given data set
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import math

df = pd. read_csv ('A2Q1.csv',header = None)  # Importing excel data file

XT = np.array(df)
X = np.transpose(XT)

#....... The probabilistic mixture for this given data is EXPONENTIAL MIXTURE........

# Parameters beta and Pi
beta=np.zeros(4)
pi=np.zeros(4)


beta_prev = np.random.rand(4)*10
pi_prev = np.random.rand(4)*10
pi_prev = pi_prev/np.sum(pi_prev)
lamda = np.zeros((4,1000))
den = np.zeros(1000)

e = 0.01                                    # epsilon value e=0.01
dif = 10

t=1
while dif>e :                               # loop ends when norm of difference between two successive parameters less than epsilon
    beta = np.zeros(4)
    pi = np.zeros(4)
    den = np.zeros(1000)

    # Finding lamda
    for i in range(0, 1000):
        for l in range(0, 4):
            den[i] += math.exp(-X[:,i]/beta_prev[l]) * pi_prev[l]/beta_prev[l]
    for k in range(0,4):
        for i in range(0,1000):
            lamda[k][i] =math.exp(-X[:,i]/beta_prev[k])*pi_prev[k]/beta_prev[k]
            lamda[k][i]=lamda[k][i]/den[i]


    # Finding parameter beta
    for k in range(0,4):
        num_beta = 0
        den_beta = 0
        for i in range(0,1000):
            num_beta += lamda[k][i]*X[:,i]
            den_beta += lamda[k][i]
        beta[k] = num_beta/den_beta

    # Finding parameter pi
    for k in range(0, 4):
        num_pi = 0
        for i in range(0, 1000):
            num_pi += lamda[k][i]
        pi[k] = num_pi/1000

    # Finding log of maximum likelihood function
    log_max_l = 0
    for i in range(0,1000):
        sum=0
        for k in range(0,4):
            max_l = math.exp(-X[:,i]/beta[k])*pi[k]/beta[k]
            sum += max_l
        log_max_l += math.log(sum)

    mp.plot(t,log_max_l,'ro')                      # Plotting number of iterations vs log of maximum likelihood function


    dif = (np.linalg.norm(beta_prev-beta)**2 + np.linalg.norm(pi_prev-pi)**2)**0.5       # norm of difference between two successive parameters
    t+=1


    beta_prev = beta
    pi_prev = pi
    
mp.title('EM algorithm for Exponential mixture')            # Giving Labels
mp.xlabel(' number of iterations ')                         # Labeling X axis
mp.ylabel(' log of maximum likelihood function')            # Labeling Y axis
mp.show()