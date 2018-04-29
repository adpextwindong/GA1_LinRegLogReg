import sys
import datetime
import numpy as np
import scipy
import pandas
import csv
import math
import collections
import bisect


import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages

#Decision tree class 
#If dec is None, the node is a leaf node 
class DTree(object):
    def __init__(self):
        self.le = None          #Subtree for less than decision
        self.ge = None          #Subtree for greater than decision
        self.dec = None         #Decision for this step of tree
        self.prediction = None  #Predecition if a leaf node


#A namedtuple to represent a continuous threshold decision2
Decision = collections.namedtuple('Decision', ['feature', 'threshold'])


#Build a decision tree for a data set
#Desicions are selected by maximization of entropy based information gain
def buildTree(X, y, depth):
  print("Building tree. Depth left: " + str(depth))
  #print("Current data set: ", X, y)
  t = DTree()
  #Base case: partition is uniform
  if (sameLabel(y)):
    print("Same label termination")
    t.prediction = y[0,0]
    print("Prediction: " + str(t.prediction))
    return t

  #Base case: depth 0
  if (depth == 0):
    print("Max depth reached termination")
    t.prediction = majority(y)
    print("Prediction: " + str(t.prediction))
    return t

  print("Finding best decision")
  t.dec = bestDecision(X, y)

  #Sort X and y by ascending values of the decision feature
  #print("Sorting X and y by decision")
  s_idx = np.argsort( np.ravel(X[:,t.dec.feature]) )
  X_sort = X[s_idx]
  y_sort = y[s_idx]

  #Partition X and y by the decision
  #print("Partitioning X and y by decision")
  part_idx = bisect.bisect_right( np.ravel(X_sort[:,t.dec.feature]), t.dec.threshold)
  print("Partition index: " + str(part_idx))

  #print("X, y less than")
  X_le = X_sort[:part_idx]
  y_le = y_sort[:part_idx]
  #print("LE size ", X_le.shape, y_le.shape)
  #print(X_le, y_le)

  #print("X, y greater than")
  X_ge = X_sort[part_idx:]
  y_ge = y_sort[part_idx:]
  #print("GE size ", X_ge.shape, y_ge.shape)
  #print(X_ge, y_ge)

  print("Building left subtree")
  t.le = buildTree(X_le, y_le, depth-1)
  print("Building right subtree")
  t.ge = buildTree(X_ge, y_ge, depth-1)

  #print("Stepping up")
  return t

def bestDecision(X, y):
  #print(y)
  best_gain = 0.0
  best_dec = None
  #For each column (feature)
  for col in range(X.shape[1]):
    #Sort X and y by ascending values of that column
    #print("Testing feature " + str(col))
    #print(X[:,col])
    #Get sorting indexes from a flattened column
    s_ind = np.argsort( np.ravel(X[:,col]) )
    #print("Sort index vector ", s_ind)
    X_sort = X[s_ind]
    y_sort = y[s_ind]
    #print("Sort arranged y", y_sort)

    #For each pair of adjacent values in the column
    for row  in range(X_sort.shape[0]-1):
      #print("Checking pair at row " + str(row))
      if (y_sort[row, 0] != y_sort[row+1, 0]):
        #print("Found pair to test at row" + str(row))
        thr = (X_sort[row, col] + X_sort[row+1, col]) / 2.0
        dec = Decision(feature=col, threshold=thr)
        gain = informationGain(X_sort, y_sort, dec)
        #print("Considering decision: ", dec, "with gain: " + str(gain))

        if (gain > best_gain):
          best_gain = gain
          best_dec = dec

  print("Best decision: ", best_dec, "With gain: " + str(best_gain))
  return best_dec
  

def majority(y):
  countp = 0
  countn = 0
  for i in range(y.size):
    if (y[i,0] == 1):
      countp += 1;
    else:
      countn += 1;

  if (countp > countn):
    return 1
  else:
    return -1

def sameLabel(y):
  for i in range(y.size):
    if (y[i,0] != y[0,0]):
      return False

  return True

def informationGain(X, y, dec):
  # get entropy ofnode, then both children and calc total gain
  part_idx = bisect.bisect_right( np.ravel(X[:,dec.feature]), dec.threshold)

  #Partition the range by the decision
  y_le = y[:part_idx]
  y_ge = y[part_idx:]
  #print(y_le)
  #print(y_ge)

  #Calculate p1 and p2 (the probability of choosing each branch)
  p1 = float(part_idx) / y.size
  p2 = 1 - p1
 
  #Calculate the entropy of the parent and child nodes
  parentEnt = getEntropy(y)
  lChildEnt = getEntropy(y_le)
  rChildEnt = getEntropy(y_ge)

  #return the resulting info gainz
  return (parentEnt-(p1 * lChildEnt + p2 * rChildEnt))


def getEntropy(y):
  #Record the total number of data points
  total = float(y.size)

  #Count the negative and positive examples
  pos = 0
  neg = 0
  for i in range(y.size):
    if (y[i,0] == 1):
      pos += 1
    else:
      neg += 1

  #Calculate the positive and negative probabilities
  p_pos = pos / total
  p_neg = neg / total

  #Calculate each part of the entropy
  if (p_pos > 0):
    ent_pos = (-1)*(p_pos * math.log(p_pos, 2))
  else:
    ent_pos = 0

  if (p_neg > 0):
    ent_neg = (-1)*(p_neg * math.log(p_neg, 2))
  else:
    ent_neg = 0

  return ent_pos + ent_neg

#def getEntropy(prob1,prob2):
#  return (-1 * prob1 * np.log2(prob1)) - (prob2 * np.log2(prob2))


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

  #print(X_train)
  #print(y_train)

  #norm = findNormalizationVector(X_train)

  #Build stump decision tree
  d = buildTree(X_train, y_train, depth=1)

if __name__ == "__main__":
  main()
