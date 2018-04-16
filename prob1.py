import sys
import datetime
import numpy as np
import scipy
import pandas
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():
	np.set_printoptions(suppress=True)
	#np.set_printoptions(precision=4)


	### Problem 1.1
	printbold("Problem 1.1")

	#Load data into X and y
	X_train = loadX('data/housing_train.txt', dummy=True)
	y_train = loady('data/housing_train.txt')

	#Calculate w, the optimal weight vector
	w = getw(X_train, y_train);
	print("Optimal weight vector, w:\n{}\n".format(w))
	### Problem 1.2
	printbold("Problem 1.2")

	ASE_train = getASE(X_train, y_train, w)

	X_test = loadX('data/housing_test.txt', dummy=True)
	y_test = loady('data/housing_test.txt')

	ASE_test = getASE(X_test, y_test, w)

	print("Training ASE: " + str(ASE_train))
	print("Testing ASE: " + str(ASE_test))
	print

	### Problem 1.3
	printbold("Problem 1.3")
	#Dummy Variable-less version of 1.1, 1.2

	X_train_no_dummy = loadX('data/housing_train.txt', dummy=False)
	X_test_no_dummy	= loadX('data/housing_test.txt', dummy=False)

	WEIGHTS_no_dummy	= getw(X_train_no_dummy, y_train)
	ASE_train_no_dummy	= getASE(X_train_no_dummy, y_train, WEIGHTS_no_dummy)
	ASE_test_no_dummy	= getASE(X_test_no_dummy, y_test, WEIGHTS_no_dummy)

	print("Dummy Variable-less training results:")
	print("Optimal weight vector, w:\n{}\n".format(WEIGHTS_no_dummy))
	print("Training ASE: " + str(ASE_train_no_dummy))
	print("Testing ASE: " + str(ASE_test_no_dummy))

	### Problem 1.4
	printbold("Problem 1.4")

	y_train_d	= loady('data/housing_train.txt')
	y_test_d	= loady('data/housing_test.txt')

	ASE_train_plot_data = []
	ASE_test_plot_data = []
	
	d_list = [0] + [ x for x in xrange(100) if x % 2 == 0]
	for d in d_list:
		X_train_d	= loadX('data/housing_train.txt', dummy=True, randFeatures=d)
		X_test_d	= loadX('data/housing_test.txt', dummy=True, randFeatures=d)

		WEIGHTS_d	= getw(X_train_d, y_train_d)
		ASE_train_d	= getASE(X_train_d, y_train_d, WEIGHTS_d)
		ASE_test_d	= getASE(X_test_d, y_test_d, WEIGHTS_d)

		#print("Rand feature addtion training results D = {}:".format(d))
		#print("Optimal weight vector, w:\n{}\n".format(WEIGHTS_d))
		#print("Training ASE: " + str(ASE_train_d))
		#print("Testing ASE: " + str(ASE_test_d))

		ASE_train_plot_data.append(ASE_train_d)
		ASE_test_plot_data.append(ASE_test_d)

	fig = plt.figure()
	plt.xlabel('d Random Features')
	plt.ylabel('ASE(d)')
	plt.plot(d_list, ASE_train_plot_data, label='train')
	plt.plot(d_list, ASE_test_plot_data, label='test')
	plt.legend(loc='upper left')
	plt.show()
	fig.savefig("1_4_Report.pdf")

def printbold(text):
	print("\033[1m" + text + "\033[0m")

def getw(X, y):
	"""Calculate the optimal weight vector for a data set"""
	w = (X.T * X).I * X.T * y

	return w


def getASE(X, y, w):
	"""Calculate the Sum Squared Error of a data & weight vector pair"""
	#Calculate the Sum Squared Error
	#Matrix multiplication produces a 1x1 matrix
	#[0,0] at the end gives the actual number
	# (at position 0,0 in the matrix)
	SSE = ((y - X*w).T * (y - X*w))[0,0]

	#Find the numver of data examples
	num_examples = y.size

	#Calculate the Average Squared Error
	ASE = SSE / num_examples

	return ASE


def loadX(fname, dummy=True, randFeatures = 0):
	"""Load independent variable data from a file"""
	X = []
	with open(fname, 'r') as datafile:
		datareader = csv.reader(datafile, delimiter=' ',
			skipinitialspace=True)
		for row in datareader:
			#Load independent and dummy variables into X
			if dummy:
				#Dummy variable in the first column
				X.append([1] + row[:-1] + np.random.standard_normal(randFeatures).tolist())
			else:
				X.append(row[:-1] + np.random.standard_normal(randFeatures).tolist())

	#Convert to numpy matrix type
	#and cast data from string to float
	X = np.matrix(X, dtype=float)

	return X


def loady(fname):
	"""Load dependent variable data from a file"""
	y = []
	with open(fname, 'r') as datafile:
		datareader = csv.reader(datafile, delimiter=' ',
			skipinitialspace=True)
		for row in datareader:
			#Load dependent variable into Y
			#Appended as list (instead of number) to make column vector
			y.append([row[-1]])

	#Convert to numpy matrix type
	#and cast data from string to float
	y = np.matrix(y, dtype=float)

	return y


if __name__ == "__main__":
	main()
