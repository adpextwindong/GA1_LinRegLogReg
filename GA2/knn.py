import sys
import datetime
import numpy as np
import scipy
import pandas
import csv
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():
	np.set_printoptions(suppress=True)
	#np.set_printoptions(precision=4)


	### Problem 1.1
	printbold("Problem 1.1")

	#TODO FILEIO
	#TODO NORMALIZATION
	#TODO KNN(K) -> [k nearest neighbors]
	#TODO CLASSIFY([NEIGHBORS]) -> CLASS LABEL
	#PLOT THESE AS FUNCTION OF K
	#TODO PLOTTING TRAINING ERROR (# of mistakes on train)
	#TODO PLOTTING TEST ERROR (# of mistakes on test)
	#TODO Leave one out crossvalidation on training set

def printbold(text):
	print("\033[1m" + text + "\033[0m")


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

def euclideanDistance(v,w):
	assert(len(v) == len(w))
	sum = 0
	for p in zip(v,w):
		sum += pow((p[0] - p[1]), 2)
	return math.sqrt(sum)

if __name__ == '__main__':
	v = [1,2,3,4]
	w = [4,3,5,7]

	data = [
		[2,3,4,5,6],
		[2,3,4,5,6],
		[7,8,5,7,3],
		[2,3,4,5,6],
		[1,2,3,4,5]
	]
	print euclideanDistance(v,w)	


