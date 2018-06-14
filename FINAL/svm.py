from sklearn import svm
import csv
import pandas as pd
import numpy as np

def main():
  data = generate_subsequences(load_data('data/Ind1_train/Subject_2_part1.csv'))
  
  k = 5

  spl_idx = []
 
  for i in range(k):
    l = len(data) * 1.0
    spl_idx.append([(int)(l * i / k), (int)(l * (i+1) /k)])

  for i in range(k):
    print("val run : " + str(i+1))
    st = spl_idx[i][0]
    end = spl_idx[i][1]
    X = data.values[:, :-1]
    y = data.values[:, -1]

    rows_idx = []
    for i in range(st):
      rows_idx.append(i)
    for i in range(len(data)-end):
      rows_idx.append(i+end)

    # we create 20 points
    X_train = X[rows_idx,:]
    y_train = y[rows_idx]

    X_valid = X[st:end,:]
    y_valid = y[st:end]

    # fit the model
    clf = svm.SVR()
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_valid)

    for thr in [x / 100.0 for x in range(10,20)]:
      res = {'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0}
      for i in range(len(X_valid)):
	predict = predictions[i]
	actual = y_valid[i]
	#print(predict, actual)
	if (actual > 0.5):
	  if (predict > thr):
	    res['TP'] += 1
	  else:
	    res['FN'] += 1
	else:
	  if (predict > thr):
	    res['FP'] += 1
	  else:
	    res['TN'] += 1
      print("Prediction threshold: " + str(thr))
      print(res)
      print("F-measure: " + str((2.0*res['TP'])/(2*res['TP']+res['FN']+res['FP'])))
        
    
  
  #clf_weights.fit(X, y, sample_weight=sample_weight_memes)



def load_data(fname):
  """Load data from a file"""
  csv_header = csv_header_train()
  
  data = pd.read_csv(fname, header=None, names=csv_header, parse_dates=['time'], infer_datetime_format=True)
  
  return data

#def load_data(fname):
#  data = []
#  with open(fname, 'r') as datafile:
#    datareader = csv.reader(datafile)
#    for row in datareader:
#        parsed_row = []
#        
#        data.append(row)
#
#  return data

def generate_subsequences(data):
  chunkidxs = []
  idx = 0
  for i in range(len(data['hypo30m'])):
    chunkidxs.append(idx)
    if (data['hypo30m'][i] == 1):
      idx += 1

  numchunks = chunkidxs[-1]

  data['chunk'] = chunkidxs
  #data.set_index('chunk')

  chunks = data.groupby('chunk')
  data_filt = chunks.filter(lambda x: len(x) >= 7)
  newchunks = data_filt.groupby('chunk')

  instances = []
  my_list = data_filt['chunk'].values
  uniqueVals = np.unique(my_list)


  for gn in uniqueVals:
    idxs = newchunks.groups[gn]
    mask = [False]*(len(data))
    for i in idxs:
      mask[i] = True
    df = data[mask]
    nd = df.values
    for i in range(len(nd)-6):
      d = nd[i:i+7,:-2].flatten('F')
      l = nd[i+6,-2]
      row = np.append(d, l)
      for i in range(7):
        row[i] = row[i].hour
      instances.append(row)
  
  df_instances = pd.DataFrame(instances, columns=flat_header_train())

  return df_instances

def flat_header_train():
  head = csv_header_test()
  head.append('hypo30m')
  return head

def csv_header_train():
  csv_header = ['time','glucose','slope','iob','mob','morning','afternoon','evening','night', 'hypo30m']

  return csv_header

def csv_header_test():
  csv_header = []
  for label in ['time','glucose','slope','iob','mob','morning','afternoon','evening','night']:
      for i in range(7):
	  csv_header.append(
	      't' + str(i) + '_' + label
	  )

  return csv_header

if __name__ == "__main__":
  main()
