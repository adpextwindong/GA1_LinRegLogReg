import os, glob

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import namedtuple

CifarData = namedtuple('CifarData', ['Activation','Epochs','Learning_Rate', 'Loss', 'Accuracy'])

the_data = []
for fname in glob.glob(os.getcwd() + '/*out.txt'):
    print fname
    details = fname.split('/')
    details.reverse()
    ACTIVATION, EPOCHS, LEARNING_RATE, _ = details[0].strip('out.txt').split('_')
    with open(fname, 'r') as f:
        lines = f.readlines()
        LOSS = eval(lines[len(lines) - 2].strip('\n'))
        ACC = eval(lines[len(lines) - 1].strip('\n'))
        
        the_data.append(CifarData(ACTIVATION, EPOCHS, LEARNING_RATE, LOSS, ACC))

print the_data