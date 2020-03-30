#!/bin/bash

function test {
    unbuffer python hyperParameterSearcher.py --features datasets/6_select_120_fromModel_no_other.csv --meta datasets/5_downsampled_100_no_other.csv --N_JOBS 20 --cache 10000 --clf $1 --mergedLabels --nIteration $2 2>&1 | tee results/hyperResults_f1_micro_$1_$2.txt
}


test RF 1000
test DT 1000
test NaiveBayes 1000
test SVMRbf 100
test SVMPoly 100
