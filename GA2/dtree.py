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

def loadX(fname):
  """Load independent variable data from a file"""
  X = []
  with open(fname, 'r') as datafile:
    datareader = csv.reader(datafile, delimiter=',',
      skipinitialspace=True)
    for row in datareader:
      #Load independent and dummy variables into X
      X.append(row[1:])

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
      y.append([row[0]])

  #Convert to numpy matrix type
  #and cast data from string to float
  y = np.matrix(y, dtype=int)

  return y

def main():
  np.set_printoptions(suppress=True)

  X_train = loadX('data/knn_train.csv')
  y_train = loady('data/knn_train.csv')

  print(X_train)
  print(y_train)


if __name__ == "__main__":
  main()
