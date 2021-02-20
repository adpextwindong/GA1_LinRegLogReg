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
	y_train = loady('data/knn_train.csv')
	test = loadX('data/knn_test.csv')
	y_test = loady('data/knn_test.csv')

	#norm_vector = findNormalizationVector(np.concatenate((train,test)))
	norm_vector = findNormalizationVector(train) 	

	normalized_train = applyNormMatrix(train, norm_vector)
	normalized_test = applyNormMatrix(test, norm_vector)

	#print normalized_train

	#print findNormalizationVector(train)
	### Problem 1.1
	printbold("Problem 1.1")
	k = 5
	random_point = applyNormVector(rand_vec(30), norm_vector)
	knn_res = knn_classify(normalized_train, y_train, random_point, k)
	#print "k: {} random_point: {}\n".format(k, random_point)
	if (knn_res == 1):
		print "K = {} : Positive Result".format(k)
	else:
		print "K = {} : Negative Result".format(k)

	#total_rows = train.shape[0]
	ROW_INDS = filter(lambda x: x%2 == 1, range(1,52))
	
	LEAVE_ONE_OUT_ACC_DATA = []
	TRAIN_ERROR = []
	TEST_ERROR = []
	print "Doing reporting"
	for k_val in ROW_INDS:
		print "Reporting on k : {}".format(k_val)
		LEAVE_ONE_OUT_ACC_DATA.append(cross_validation_leave_one(train, y_train, k_val))
		TRAIN_ERROR.append(find_train_error(normalized_train, y_train, k_val))
		TEST_ERROR.append(find_test_error(normalized_test, normalized_train, y_test, y_train, k_val))

	fig = plt.figure()
	plt.xlabel('K')
	plt.ylabel('Accuracy')
	plt.plot(ROW_INDS, TRAIN_ERROR, label='train')
	plt.plot(ROW_INDS, TEST_ERROR, label='test')
	plt.plot(ROW_INDS, LEAVE_ONE_OUT_ACC_DATA, label='leave_one_out')
	plt.legend(loc='upper right')
	plt.show()
	fig.savefig("1_2_Report.png")


#TODO Make KNN Model class so I don't have to lug around normalized_train and y_train when doing test error
def find_test_error(normalized_test, normalized_train, y_test, y_train, k_val):
	num_mistakes = 0.0
	total_rows = float(normalized_test.shape[0])
	for i in xrange(normalized_test.shape[0] - 1):
		correct_label = y_test[i,0]
		point = normalized_test[i,:].tolist()[0]

		knn_res = knn_classify(normalized_train, y_train, point, k_val)

		if(correct_label != knn_res):
			num_mistakes += 1
	
	return 1 - (num_mistakes / total_rows)

"""
	Expects normalized data
"""
def find_train_error(data, labels, k):
	num_mistakes = 0.0
	total_rows = float(data.shape[0])
	for i in xrange(data.shape[0] - 1):
		correct_label = labels[i,0]
		point = data[i,:].tolist()[0]

		knn_res = knn_classify(data, labels, point, k)

		if(correct_label != knn_res):
			num_mistakes += 1
	
	return 1 - (num_mistakes / total_rows)

"""
	Expects normalized data
"""
def cross_validation_leave_one(train_data, labels, k):
	num_mistakes = 0.0
	total_rows = float(train_data.shape[0])
	for i in xrange(train_data.shape[0] - 1):
		correct_label = labels[i,0]
		point = train_data[i,:].tolist()[0]

		#Leave one out process
		left_one_out_data, left_one_out_labels = leave_one_out(train_data, labels, i)

		knn_res = knn_classify(left_one_out_data, left_one_out_labels, point, k)

		if(correct_label != knn_res):
			num_mistakes += 1
	
	return 1 - (num_mistakes / total_rows)

def leave_one_out(data_matrix, labels, index):
	ret_data = np.delete(data_matrix.copy(), (index), axis=0)
	ret_labels = np.delete(labels.copy(), (index), axis=0)

	return (ret_data, ret_labels)

"""
	Expects normalized data, returns -1 or 1 classifier
"""
def knn_classify(data, labels, point, k):
	neighbors = naive_knn(data, point,k)

	classes = []
	for _, neighbor_i in neighbors:
		classes.append(labels[neighbor_i,0])
		#print labels[neighbor_i,0]
	positives = filter(lambda x: x == 1,classes)
	negatives = filter(lambda x: x == -1, classes)

	if(len(positives) >= len(negatives)):
		return 1
	else:
		return -1


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

def euclideanDistance(v,w):
	assert(len(v) == len(w))
	sum = 0
	for p in zip(v,w):
		sum += pow((p[0] - p[1]), 2)
	return math.sqrt(sum)

def naive_knn(data, point, k):
	dists = []
	A = np.squeeze(np.asarray(data)).tolist()
	for i,p in enumerate(A):
		dists.append((euclideanDistance(point, p), i))
	neighbors_val_ind_pairs = sorted(dists, key=lambda x : x[0])
	#print neighbors_val_ind_pairs
	ret = []
	for _, i in neighbors_val_ind_pairs[:k]:
		ret.append((data[i,:].tolist()[0],i))

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
