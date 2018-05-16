#!/bin/sh

echo 'DROPOUT 0.2'
echo 'Starting Sig testing'
echo 'SIG 10 0.01'
echo `date`
python cifar_p3.py 1 SIG 10 0.01 0.2 0.5 > 1_SIG_10_0.01_0.2_0.5_out.txt

echo 'SIG 10 0.001'
echo `date`
python cifar_p3.py 1 SIG 10 0.001 0.2 0.5 > 1_SIG_10_0.001_0.2_0.5_out.txt

echo 'Starting ReLu testing'
echo 'RELU 10 0.01'
echo `date`
python cifar_p3.py 1 RELU 10 0.01 0.2 0.5 > 1_RELU_10_0.01_0.2_0.5_out.txt

echo 'RELU 10 0.001'
echo `date`
python cifar_p3.py 1 RELU 10 0.001 0.2 0.5 > 1_RELU_10_0.001_0.2_0.5_out.txt

echo 'DROPOUT 0.3'

echo 'Starting Sig testing'
echo 'SIG 10 0.01'
echo `date`
python cifar_p3.py 1 SIG 10 0.01 0.2 0.5 > 1_SIG_10_0.01_0.2_0.5_out.txt

echo 'SIG 10 0.001'
echo `date`
python cifar_p3.py 1 SIG 10 0.001 0.2 0.5 > 1_SIG_10_0.001_0.2_0.5_out.txt

echo 'Starting ReLu testing'
echo 'RELU 10 0.01'
echo `date`
python cifar_p3.py 1 RELU 10 0.01 0.2 0.5 > 1_RELU_10_0.01_0.2_0.5_out.txt

echo 'RELU 10 0.001'
echo `date`
python cifar_p3.py 1 RELU 10 0.001 0.2 0.5 > 1_RELU_10_0.001_0.2_0.5_out.txt