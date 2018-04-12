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

	print(w)
				


if __name__ == "__main__":
	main()
