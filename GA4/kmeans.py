import sys
import csv
import numpy as np
import scipy
import random

import matplotlib.pyplot as plt
import seaborn as sns

normsquare = lambda x: np.inner(x, x)
PLOT = True
PRINT = False

def main():
  #set random number generator seed
  np.random.seed()
  #sns.set_style("darkgrid")

  ### Problem 2.1
  if (PRINT):
    print("~~~~ Problem 2.1 ~~~~")

  #Load data into X
  X = loadX("data-1.txt")

  part, p1SSEs = kmeans(X, k=2, logSSE=True)
  if (PLOT):
    fig1 = plt.figure()
    plt.plot(range(1, len(p1SSEs)+1), p1SSEs) 
    plt.ylabel('SSE')
    plt.xlabel('Iteration')
    plt.title('K-means Convergence for k=2')
    plt.show()
    fig1.savefig("2-1_Report.png")

  ### Problem 2.2
  if (PRINT):
    print("~~~~ Problem 2.2 ~~~~")


  minSSEs = []
  for k in range(2,11):
    if (PRINT):
      print("k = " + str(k))
    SSEs = []
    for i in range(10):
      thisSSE = getSSE(kmeans(X, k))
      if (PRINT):
        print("Iteration " + str(i+1) + " SSE: " + str(thisSSE))
      SSEs.append(thisSSE)

    if (PRINT):
      print("Min SSE: " + str(min(SSEs)))
    minSSEs.append(min(SSEs))

  if (PLOT):
    fig2 = plt.figure()
    plt.plot(range(2, 11), minSSEs) 
    plt.ylabel('SSE')
    plt.xlabel('Value of k')
    plt.title('K-means SSE with Varying k (Minimum SSE of 10 runs per k)')
    plt.show()
    fig2.savefig("2-2_Report.png")

def kmeans(X, k, logSSE=False):
  """Use kmeans to split X into k partitions"""
  if (k > len(X) or k <= 1):
    return X


  means = [X[i] for i in random.sample(range(len(X)), k)]
  SSE = 0
  step = 0
  SSEs = []

  while True:
    step += 1

    parts = [[] for i in range(k)]
    for vec in X:
      norms = [normsquare(vec - m) for m in means]
      idx = np.argmin(norms)
      parts[idx].append(vec)
    means = getMeans(parts)

    lastSSE = SSE
    SSE = getSSE(parts, means)

    if (logSSE):
      SSEs.append(SSE)
      if (PRINT):
	print("Iteration: " + str(step))
	print("SSE: " + str(SSE))
	print

    if (SSE == lastSSE):
      break

  if (logSSE):
    if (PRINT):
      print("Converged")
    return (parts, SSEs)

  return parts 

def getMeans(parts):
  """Calculate the mean vector for each partition"""
  means = []
  for i in range(len(parts)):
    means.append(np.mean(parts[i], axis=0))

  return means
    

def getSSE(parts, means=None):
  """Calculate the Sum Squared Error of a set of partitions"""

  #Calculate means if not given
  if means is None:
    means = getMeans(parts)

  #Calculate the Sum Squared Error
  SSE = 0
  for i in range(len(parts)):
    for j in range(len(parts[i])):
      diff = parts[i][j] - means[i]
      SSE += normsquare(diff)

  return SSE

def loadX(fname):
  """Load data from a file"""
  X = []
  with open(fname, 'r') as datafile:
    datareader = csv.reader(datafile)
    for row in datareader:
      #Load independent and dummy variables into X
      X.append(row)

  #Convert to numpy matrix type
  #and cast data from string to float
  X = np.array(X, dtype=int)

  return X

if __name__ == "__main__":
  main()  
