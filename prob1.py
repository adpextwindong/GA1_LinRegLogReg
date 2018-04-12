import sys
import datetime
import numpy as np
import scipy
import pandas
import csv

import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages

def main():

  np.set_printoptions(suppress=True)
  #np.set_printoptions(precision=4)

  ### Problem 1.1

  #Load data into X and y
  X = []
  y = []
  with open('data/housing_train.txt', 'r') as datafile:
    datareader = csv.reader(datafile, delimiter=' ',
      skipinitialspace=True)
    for row in datareader:
      #Load independent and dummy variables into X
      #Dummy variable in the first column
      X.append([1] + row[:-1])
      #Load dependent variable into Y
      #Appended as list (instead of number) to make column vector
      y.append([row[-1]])

  #Convert to numpy matrix types
  #and cast data from string to float
  X = np.matrix(X, dtype=float)
  y = np.matrix(y, dtype=float)

  #Calculate w, the optimal weight vector
  w = (X.T * X).I * X.T * y
  print("Optimal weight vector, w:")
  print(w)
  print

  ### Problem 1.2
  
  #Calculate the Sum Squared Error
  #Matrix multiplication produces a 1x1 matrix
  #[0,0] at the end gives the actual number
  # (at position 0,0 in the matrix)
  SSE = ((y - X*w).T * (y - X*w))[0,0]

  #Find the numver of data examples
  num_examples = y.size

  #Calculate the Average Squared Error
  ASE = SSE / num_examples
  print("Training ASE:")
  print(ASE)
  print
  

if __name__ == "__main__":
  main()
