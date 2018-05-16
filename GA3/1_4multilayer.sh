#!/bin/sh

echo '2 Hidden Layer'
echo 'Starting Sig testing'
echo 'SIG 10 0.01'
echo `date`
python cifar_p3.py 2 SIG 10 0.01 0.2 0.5 > 2_SIG_10_0.01_0.2_0.5_out.txt

echo 'SIG 10 0.001'
echo `date`
python cifar_p3.py 2 SIG 10 0.001 0.2 0.5 > 2_SIG_10_0.001_0.2_0.5_out.txt

echo 'Starting ReLu testing'
echo 'RELU 10 0.01'
echo `date`
python cifar_p3.py 2 RELU 10 0.01 0.2 0.5 > 2_RELU_10_0.01_0.2_0.5_out.txt

echo 'RELU 10 0.001'
echo `date`
python cifar_p3.py 2 RELU 10 0.001 0.2 0.5 > 2_RELU_10_0.001_0.2_0.5_out.txt

