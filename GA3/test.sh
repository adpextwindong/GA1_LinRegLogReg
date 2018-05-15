#!/bin/sh

python cifar_p3.py 10 0.1 > sig_10_0.1_out.txt
python cifar_p3.py 10 0.01 > sig_10_0.01_out.txt
python cifar_p3.py 10 0.001 > sig_10_0.001_out.txt
python cifar_p3.py 10 0.0001 > sig_10_0.0001_out.txt
