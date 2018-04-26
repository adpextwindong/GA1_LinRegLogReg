import sys
import datetime

import numpy as np
import scipy
import pandas

import csv

import math
from random import randint

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():
	np.set_printoptions(suppress=True)
	#np.set_printoptions(precision=4)

	train = loadX('data/knn_train.csv')
	test = loadX('data/knn_test.csv')

	norm_vector = findNormalizationVector(np.concatenate((train,test)))
	
	normalized_train = applyNorm(train, norm_vector)
	normalized_test = applyNorm(test, norm_vector)

	print normalized_train

	#print findNormalizationVector(train)
	### Problem 1.1
	printbold("Problem 1.1")
	
	#TODO CLASSIFY([NEIGHBORS]) -> CLASS LABEL
	#PLOT THESE AS FUNCTION OF K
	#TODO PLOTTING TRAINING ERROR (# of mistakes on train)
	#TODO PLOTTING TEST ERROR (# of mistakes on test)
	#TODO Leave one out crossvalidation on training set

def printbold(text):
	print("\033[1m" + text + "\033[0m")

### returns a copy with the norm applied
def applyNorm(in_matrix, norm_vector):
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

def euclideanDistance(v,w):
	assert(len(v) == len(w))
	sum = 0
	for p in zip(v,w):
		sum += pow((p[0] - p[1]), 2)
	return math.sqrt(sum)

def naive_knn(data, point, k):
	dists = []
	for i,p in enumerate(data):
		dists.append((euclideanDistance(point, p), i))
	neighbors_val_ind_pairs = sorted(dists, key=lambda x : x[0])
	#print neighbors_val_ind_pairs
	ret = []
	for _, i in neighbors_val_ind_pairs:
		ret.append(data[i])
	return ret

def rand_vec(dimension):
	return [randint(0, 300) for _ in xrange(dimension)]

def rand_dataset(n_data_points, dimension):
	return [rand_vec(dimension) for _ in xrange(n_data_points)]

def benchmark_naive_knn(scale_d, scale_n, k):
	scale_test_p = rand_vec(scale_d)
	scale_test_data = rand_dataset(scale_n, scale_d)
	naive_knn(scale_test_data, scale_test_p, k)

if __name__ == '__main__':
	main()
