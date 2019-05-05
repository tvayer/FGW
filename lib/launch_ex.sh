#!/bin/sh

data_path="../data"
results_path="../results"

python3 nested_cv_fgw.py -dn ptc -r $results_path -ni 10 -no 50 -d $data_path -fea hamming_dist -st shortest_path -cva True -wl 2 -am True&









