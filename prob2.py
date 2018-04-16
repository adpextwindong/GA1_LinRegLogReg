import sys
import datetime
import numpy as np
import scipy
import pandas
import csv
import math

import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages

def main():
  np.set_printoptions(suppress=True)
  #np.set_printoptions(precision=4)


  ### Problem 2.1
  printbold("Problem 2.1")

  #Load training data
  X_train = loadX('data/usps-4-9-train.csv', dummy=True)
  y_train = loady('data/usps-4-9-train.csv')

  #Load testing data
  X_test = loadX('data/usps-4-9-test.csv', dummy=True)
  y_test = loady('data/usps-4-9-test.csv')
  #Adjust the values of X so x_ij is in [0,1]
  X_train = X_train * (1.0/256)

  #Create a gradient function for the first section
  p2_1grad = lambda pos: logisticgradient(X_train, y_train, pos)

  w_0 = np.matrix([0] * X_train.shape[1]).T

  #Calculate w, the optimal weight vector
  w = gradientdescent(p2_1grad, eta=0.02, epsilon=0.1, w_0=w_0) 

  print("Optimal weight vector, w:")
  print(w)
  print

  accuracy_train = accuracy(X_train, y_train, w)
  accuracy_test = accuracy(X_test, y_test, w)

  print("Training accuracy: " + str(accuracy_train))
  print("Testing accuracy: " + str(accuracy_train))
  print





def printbold(text):
  print("\033[1m" + text + "\033[0m")


def gradientdescent(gradf, eta, epsilon, w_0): 
  #Initialize 'w' to a column vector of zeros
  w = w_0

  while True:
    #Calculate the gradient at position 'w'
    nabla = gradf(w)
    #Descend based on the learning rate 'eta' and gradient 'nabla'
    w = w - eta * nabla

    print(np.linalg.norm(nabla))
    
    if (np.linalg.norm(nabla) < epsilon):
      #Break when stop condition has been reached
      break

  return w

def logisticgradient(X, y, w):
  #initialize nabla to a column vector of zeros
  nabla = np.matrix([0] * w.size).T

  for i in range(y.size):
    #Grab a row of X and transpose it to a column vector
    x_i = np.matrix(X[i,:]).T
    #Calculate y_hat_i
    y_hat_i = sigmoid((w.T * x_i)[0,0])
    #Update the gradient
    nabla = nabla + (y_hat_i - y[i,0]) * x_i

  return nabla


def accuracy(X, y, w):
  #Store the number of data points
  total = y.size
  #Initalize a variable for the number of correct predictions
  correct = 0

  #Calculate the number of correct predictions
  for i in range(total):
    #Grab a row of X and transpose it to a column vector
    x_i = np.matrix(X[i,:]).T
    #Calculate the decision term
    dec = (w.T * x_i)[0,0]
    if dec > 0 and y[i,0] == 1:
      correct += 1
    elif dec < 0 and y[i,0] == 0:
      correct += 1

  return correct * (1.0/total)


def logisticP1(x, w):
  return sigmoid((w.T * x)[0,0])


def logisticP0(x, w):
  return 1 - logisticP1(x, w)

def sigmoid(val):
  return 1.0/(1 + math.exp(-val))


def loadX(fname, dummy=True):
  """Load independent variable data from a file"""
  X = []
  with open(fname, 'r') as datafile:
    datareader = csv.reader(datafile, delimiter=',',
      skipinitialspace=True)
    for row in datareader:
      #Load independent and dummy variables into X
      if dummy:
        #Dummy variable in the first column
        X.append([1] + row[:-1])
      else:
        X.append(row[:-1])

  #Convert to numpy matrix type
  #and cast data from string to float
  X = np.matrix(X, dtype=float)

  return X


def loady(fname):
  """Load dependent variable data from a file"""
  y = []
  with open(fname, 'r') as datafile:
    datareader = csv.reader(datafile, delimiter=',',
      skipinitialspace=True)
    for row in datareader:
      #Load dependent variable into Y
      #Appended as list (instead of number) to make column vector
      y.append([row[-1]])

  #Convert to numpy matrix type
  #and cast data from string to float
  y = np.matrix(y, dtype=int)

  return y


if __name__ == "__main__":
  main()
