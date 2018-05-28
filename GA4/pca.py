import sys
import csv
import numpy as np
import seaborn as sns
import scipy
import random

from numpy import linalg as LA


def normsquare(x): return np.inner(x, x)


def loadX(fname):
    """Load data from a file"""
    X = []
    with open(fname, 'r') as datafile:
        datareader = csv.reader(datafile)
        for row in datareader:
            # Load independent and dummy variables into X
            X.append(row)
    # Convert to numpy matrix type
    # and cast data from string to float
    X = np.array(X, dtype=int)

    return X

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
        flatM = data_matrix.flatten()
        minval = min(flatM)
        maxval = max(flatM)
        absmax = max(abs(minval), abs(maxval))
	normRanges = [(absmax, -absmax) for col in xrange(number_of_features)]
	return normRanges

def pca(data, n):
    #Calculate mean
    mean = data.mean(axis=0)
    #print(mean.shape)
    mean_norm = np.true_divide(mean, 255)
    np.savetxt("mean.csv", mean_norm[None], fmt="%.5f", delimiter=",")

    #Construct covariance matrix
    dim = data.shape
    cov_mat = np.zeros((dim[1], dim[1]))
    for vec in data:
      diff = vec - mean
      cov_mat += np.outer(diff, diff)

    cov_mat = np.true_divide(cov_mat, dim[0])

    #data -= mean

    #cov_mat = np.cov(data, rowvar=False)
    w, v = LA.eigh(cov_mat)

    idx = np.argsort(w)[::-1]
    v = v[:,idx]
    w = w[idx]
    

    top_n = v[:,:n]
    norm_vec = findNormalizationVector(top_n)
    adj_top_n = applyNormMatrix(top_n, norm_vec)

    np.set_printoptions(suppress=True, precision=5)
    np.savetxt("eigenvectors.csv", adj_top_n.T, fmt="%.5f", delimiter=",")

    norms = LA.norm(top_n, axis=0)
    eigen_normed = top_n / norms

    return eigen_normed


if __name__ == "__main__":

    X = loadX("data-1.txt")
    e_n = pca(X,10)

    d10proj = np.dot(X, e_n)
    idxs = np.argmax(np.abs(d10proj), axis=0)
  
    images = X[idxs]

    disp_images = np.true_divide(images, 255)

    np.savetxt("3-3_images.csv", disp_images, fmt="%.5f", delimiter=",")
    

    # Problem 2.1
    #print("~~~~ Problem 3.1 ~~~~")
    #mean = X.mean(axis=0)
    #X = X - mean
    #print "Data without mean "
    #print X



    
