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

def pca(data, n):
    #Calculate mean
    mean = data.mean(axis=0)

    #Construct covariance matrix
    dim = data.shape
    cov_mat = np.zeros((dim[1], dim[1]))
    for vec in data:
      diff = vec - mean
      cov_mat += np.outer(diff, diff)

    #cov_mat = np.true_divide(cov_mat, dim[0])

    #data -= mean

    #cov_mat = np.cov(data, rowvar=False)
    w, v = LA.eigh(cov_mat)

    idx = np.argsort(w)[::-1]
    v = v[:,idx]
    w = w[idx]

    top_n_vecs = v[:,:n]
    np.savetxt("eigenvectors.csv", top_n_vecs.T, delimiter=",")

if __name__ == "__main__":

    X = loadX("data-1.txt")
    pca(X,10)
    # Problem 2.1
    print("~~~~ Problem 3.1 ~~~~")
    #mean = X.mean(axis=0)
    #X = X - mean
    print "Data without mean "
    print X



    
