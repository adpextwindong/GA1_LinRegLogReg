
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import csv
import pandas as pd



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
            't' + str(i) + '_' + label)

  return csv_header

def predict_ints(model, dataset, percentile=95):
        err_down = []
        err_up = []
        for x in range(len(dataset)):
            preds=[]
            for pred in model.estimators_:
                preds.append(pred.predict(dataset[x])[0])
                err_down.append(np.percentile(preds, (100 - percentile) / 2.))
                err_up.append(np.percentile(preds, 100 - (100-percentile) / 2.))
                return err_down, err_up

variable = load_data('data/Ind1_train/Subject_2_part1.csv')
variable = generate_subsequences(variable)

trainingDataX = variable.values[:,:-1]
trainingDataY = variable.values[:,-1]

    #split up dataset here
    #use 1000 randomized trees for some variance

rf = RandomForestRegressor(n_estimators=1000, criterion='mse', max_features='auto', bootstrap='true', oob_score='true', n_jobs =4)
    #build the regression tree
rf.fit(trainingDataX, trainingDataY)
#print(rf.feature_importances_)
prediction = rf.predict(trainingDataX)
for thr in [x / 10.0 for x in range(10)]:
    res = {'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0}
    for i in range(len(trainingDataX)):
        predict = prediction[i]
        #print(rf.predict([trainingDataX[i,:]]))
        actual = trainingDataY[i]
        
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
   #lr = sk.LogisticRegression()
    #lr.fit(trainingDataX, trainingDataY)
    #run regression on the classifier
    #svc_w_linear_kernel = SVC(kernel='linear')
#trainsize= 58 #leave out 5 for validation

#the following may not be necessary
#err_down, err_up = predict_ints(rf, trainingDataX[idx[trainsize:]], percentile=90)
#truth = trainingDataY[idx[trainsize:]]
#correct = 0.
#for i, val in enumerate(truth):
#    if err_down[i] <= val <= err_up[i]:
#        correct += 1
#print correct/len(truth)