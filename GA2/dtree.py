import sys
import datetime
import numpy as np
import scipy
import pandas
import csv
import math
import collections

import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages

class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None

Decision = namedtuple('Decision', ['feature', 'threshold'])

def printbold(text):
	print("\033[1m" + text + "\033[0m")

def applyNormVector(in_vec, norm_vec):
	out_vec = list(in_vec)
	for col,(col_min, col_max) in enumerate(norm_vec):
			col_range = col_max - col_min
			out_vec[col] = (out_vec[col] -  col_min) / col_range

	return out_vec

### returns a copy with the norm applied
def applyNormMatrix(in_matrix, norm_vector):
	ret_matrix = in_matrix.copy()
	rows = in_matrix.shape[0]
	assert(in_matrix.shape[1] == len(norm_vector))
	for row in xrange(rows):
		for col,(col_min, col_max) in enumerate(norm_vector):
			col_range = col_max - col_min
			ret_matrix[row, col] = (ret_matrix[row, col] -  col_min) / col_range
			

	return ret_matrix

def findNormalizationVector(data_matrix):
	number_of_features = data_matrix.shape[1]
	normRanges = [(np.min(data_matrix[:,col]) , (np.max(data_matrix[:,col]))) for col in xrange(number_of_features)]
	return normRanges

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
