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
    X = np.array(X, dtype=float)

    return X

def pca(data, n):
    means = data.mean(axis=0)
    data -= means

    cov_mat = np.cov(data, rowvar=False)
    w, v = LA.eigh(cov_mat)

    idx = np.argsort(w)[::-1]
    v = v[:,idx]
    w = w[idx]

    top_n_vecs = v[:n,:]
    print np.dot(v.T, X.T).T

if __name__ == "__main__":

    X = loadX("data-1.txt")
    pca(X,10)
    # Problem 2.1
    print("~~~~ Problem 23.1 ~~~~")
    #mean = X.mean(axis=0)
    #X = X - mean
    print "Data without mean "
    print X



    