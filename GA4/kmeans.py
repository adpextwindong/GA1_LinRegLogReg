import sys
import csv
import numpy as np
import scipy
import random

normsquare = lambda x: np.inner(x, x)

def main():
  #set random number generator seed
  np.random.seed()

  ### Problem 2.1
  print("~~~~ Problem 2.1 ~~~~")

  #Load data into X
  X = loadX("data-1.txt")

  part, p1SSEs = kmeans(X, k=2, logSSE=True)

  ### Problem 2.2
  print("~~~~ Problem 2.2 ~~~~")

  for k in range(2,11):
    print("k = " + str(k))
    SSEs = []
    for i in range(10):
      thisSSE = getSSE(kmeans(X, k))
      print("Iteration " + str(i+1) + " SSE: " + str(thisSSE))
      SSEs.append(thisSSE)

    print("Min SSE: " + str(min(SSEs)))


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
      print("Iteration: " + str(step))
      print("SSE: " + str(SSE))
      print

    if (SSE == lastSSE):
      break

  if (logSSE):
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
