#!/bin/sh

echo 'Starting Sig testing'
echo 'SIG 10 0.1'
echo `date`
python cifar_p3.py SIG 10 0.1	 > SIG_10_0.1_out.txt

echo 'SIG 10 0.01'
echo `date`
python cifar_p3.py SIG 10 0.01	 > SIG_10_0.01_out.txt

echo 'SIG 10 0.001'
echo `date`
python cifar_p3.py SIG 10 0.001	 > SIG_10_0.001_out.txt

echo 'SIG 10 0.0001'
echo `date`
python cifar_p3.py SIG 10 0.0001 > SIG_10_0.0001_out.txt

echo 'Starting ReLu testing'
echo 'RELU 10 0.1'
echo `date`
python cifar_p3.py RELU 10 0.1    > RELU_10_0.1_out.txt

echo 'RELU 10 0.01'
echo `date`
python cifar_p3.py RELU 10 0.01   > RELU_10_0.01_out.txt

echo 'RELU 10 0.001'
echo `date`
python cifar_p3.py RELU 10 0.001  > RELU_10_0.001_out.txt

echo 'RELU 10 0.0001'
echo `date`
python cifar_p3.py RELU 10 0.0001 > RELU_10_0.0001_out.txt
