#!/bin/sh

echo 'Momentum 0.2'
echo 'Starting Sig testing'
echo 'SIG 10 0.01'
echo `date`
python cifar_p3.py 1 SIG 10 0.01 0.2 0.01 > 1_SIG_10_0.01_0.2_0.01_out.txt

echo 'SIG 10 0.001'
echo `date`
python cifar_p3.py 1 SIG 10 0.001 0.2 0.01 > 1_SIG_10_0.001_0.2_0.01_out.txt

echo 'Starting ReLu testing'
echo 'RELU 10 0.01'
echo `date`
python cifar_p3.py 1 RELU 10 0.01 0.2 0.01 > 1_RELU_10_0.01_0.2_0.01_out.txt

echo 'Momentum 0.5'
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

echo 'Momentum 0.75'
echo 'Starting Sig testing'
echo 'SIG 10 0.01'
echo `date`
python cifar_p3.py 1 SIG 10 0.01 0.2 0.75 > 1_SIG_10_0.01_0.2_0.75_out.txt

echo 'SIG 10 0.001'
echo `date`
python cifar_p3.py 1 SIG 10 0.001 0.2 0.75 > 1_SIG_10_0.001_0.2_0.75_out.txt

echo 'Starting ReLu testing'
echo 'RELU 10 0.01'
echo `date`
python cifar_p3.py 1 RELU 10 0.01 0.2 0.75 > 1_RELU_10_0.01_0.2_0.75_out.txt

