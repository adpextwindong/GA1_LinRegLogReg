#!/bin/sh

echo 'WEGHT_DECAY 0.1'
echo 'Starting Sig testing'

echo 'SIG 10 0.01'
echo `date`
python cifar_p3.py 1 SIG 10 0.001 0.1 0.5 0.1 > 1_SIG_10_0.001_0.1_0.5_0.1_out.txt

echo 'Starting ReLu testing'
echo 'RELU 10 0.001'
echo `date`
python cifar_p3.py 1 RELU 10 0.01 0.1 0.5 0.1 > 1_RELU_10_0.01_0.1_0.5_0.1_out.txt

echo 'WEGHT_DECAY 0.25'
echo 'Starting Sig testing'
echo 'SIG 10 0.01'
echo `date`
python cifar_p3.py 1 SIG 10 0.001 0.1 0.5 0.25 > 1_SIG_10_0.001_0.1_0.5_0.25_out.txt

echo 'Starting ReLu testing'
echo 'RELU 10 0.001'
echo `date`
python cifar_p3.py 1 RELU 10 0.01 0.1 0.5 0.25 > 1_RELU_10_0.01_0.1_0.5_0.25_out.txt


echo 'WEGHT_DECAY 0.5'
echo 'Starting Sig testing'
echo 'SIG 10 0.01'
echo `date`
python cifar_p3.py 1 SIG 10 0.001 0.1 0.5 0.5 > 1_SIG_10_0.001_0.1_0.5_0.5_out.txt

echo 'Starting ReLu testing'
echo 'RELU 10 0.001'
echo `date`
python cifar_p3.py 1 RELU 10 0.01 0.1 0.5 0.5 > 1_RELU_10_0.01_0.1_0.5_0.5_out.txt

