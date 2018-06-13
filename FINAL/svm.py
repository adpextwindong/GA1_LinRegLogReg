import sklearn as sk
import csv
import pandas as pd
import numpy

def load_data(fname):
  """Load data from a file"""
  csv_header = csv_header_train()
  
  data = pd.read_csv(fname, header=None, names=csv_header, parse_dates=['time'], infer_datetime_format=True)
  
  return data

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

print(load_data('data/Ind1_train/Subject_2_part1.csv'))
