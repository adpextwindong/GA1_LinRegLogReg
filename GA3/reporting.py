import os, glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import namedtuple

CifarData = namedtuple('CifarData', ['Activation','Epochs','Learning_Rate', 'Loss_Data', 'Accuracy_Data'])

the_data = []
for fname in glob.glob(os.getcwd() + '/*out.txt'):
    print fname
    details = fname.split('/')
    details.reverse()
    ACTIVATION, EPOCHS, LEARNING_RATE, _ = details[0].strip('out.txt').split('_')
    EPOCHS = int(EPOCHS)
    LEARNING_RATE = float(LEARNING_RATE)

    with open(fname, 'r') as f:
        lines = f.readlines()
        LOSS_DATA = eval(lines[len(lines) - 2].strip('\n'))
        ACC_DATA = eval(lines[len(lines) - 1].strip('\n'))
        assert(EPOCHS == len(LOSS_DATA))

        the_data.append(CifarData(ACTIVATION, EPOCHS, LEARNING_RATE, LOSS_DATA, ACC_DATA))

print the_data

assert len(set([x.Epochs for x in the_data])) == 1
ROW_INDS = range(1,EPOCHS + 1)

fig = plt.figure()
ax = plt.subplot(111)
plt.xlabel('Epochs')
plt.ylabel('Loss')
for d in the_data:
    d_label = d.Activation + ' ' + str(d.Learning_Rate)
    plt.plot(ROW_INDS, d.Loss_Data, label=d_label)
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.175),
          fancybox=True, shadow=True, ncol=3)

plt.show()
fig.savefig("1_1_Report_Loss.png")

fig = plt.figure()
ax = plt.subplot(111)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
for d in the_data:
    d_label = d.Activation + ' ' + str(d.Learning_Rate)
    plt.plot(ROW_INDS, d.Accuracy_Data, label=d_label)
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.175),
          fancybox=True, shadow=True, ncol=3)

plt.show()
fig.savefig("1_2_Report_Loss.png")