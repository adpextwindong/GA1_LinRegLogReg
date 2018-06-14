import svm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

TEST_orig_data = svm.generate_subsequences(svm.load_data('data/Ind1_train/Subject_2_part1.csv'))
TEST_labels = TEST_orig_data['hypo30m']

def kmeans_pred(data, labels):
    csv_header = ['t0_time', 't1_time', 't2_time', 't3_time', 't4_time', 't5_time', 't6_time', 't0_glucose', 't1_glucose', 't2_glucose', 't3_glucose', 't4_glucose', 't5_glucose', 't6_glucose', 't0_slope', 't1_slope', 't2_slope', 't3_slope', 't4_slope', 't5_slope', 't6_slope', 't0_iob', 't1_iob', 't2_iob', 't3_iob', 't4_iob', 't5_iob', 't6_iob', 't0_mob', 't1_mob', 't2_mob', 't3_mob', 't4_mob', 't5_mob', 't6_mob', 't0_morning', 't1_morning', 't2_morning', 't3_morning', 't4_morning', 't5_morning', 't6_morning', 't0_afternoon', 't1_afternoon', 't2_afternoon', 't3_afternoon', 't4_afternoon', 't5_afternoon', 't6_afternoon', 't0_evening', 't1_evening', 't2_evening', 't3_evening', 't4_evening', 't5_evening', 't6_evening', 't0_night', 't1_night', 't2_night', 't3_night', 't4_night', 't5_night', 't6_night']

    rows, features = data.shape

    drop_cols = csv_header[csv_header.index('t0_morning'):]
    data = data.drop(drop_cols, inplace=False, axis=1)
    scaled_data = scale(data)
    km_estimator = KMeans(init='random', n_init=10, n_clusters = 100)

    km_estimator.fit(scaled_data)
    allocations = km_estimator.labels_
    centers = km_estimator.cluster_centers_
    PARAM_ANOMALY_THRESHOLD = 5.935120307219966 #Adapted from dist standard deviation of previous runs
    anomalies = []
    dists = []

    drop_ind = csv_header.index('t6_time')

    correct = 0
    false_positive = 0

    res = []
    for i, point in enumerate(scaled_data):
        assigned_cluster_center = centers[allocations[i]]
        diff = assigned_cluster_center[drop_ind:] - point[drop_ind:]
        dist = np.dot(diff,diff)**0.5
        if(dist == 0):
            dists.append(dist)
            res.append(0)
        elif(dist > PARAM_ANOMALY_THRESHOLD):
            anomalies.append(i)
            if(labels[i] == 1):
                correct += 1
                res.append(1)
            else:
                false_positive += 1
                res.append(0)
        else:
            res.append(0)

    return res