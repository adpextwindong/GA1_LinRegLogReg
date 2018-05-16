#!/bin/sh

python cifar_p3.py SIG 10 0.1 > sig_10_0.1_out.txt
python cifar_p3.py SIG 10 0.01 > sig_10_0.01_out.txt
python cifar_p3.py SIG 10 0.001 > sig_10_0.001_out.txt
python cifar_p3.py SIG 10 0.0001 > sig_10_0.0001_out.txt

python cifar_p3.py RELU 10 0.1 > RELU_10_0.1_out.txt
python cifar_p3.py RELU 10 0.01 > RELU_10_0.01_out.txt
python cifar_p3.py RELU 10 0.001 > RELU_10_0.001_out.txt
python cifar_p3.py RELU 10 0.0001 > RELU_10_0.0001_out.txt
