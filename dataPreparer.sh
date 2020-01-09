#!/bin/bash

# python dataPreparer.py --featuresFile datasets/elvis/2_binarized_cell_features.csv --action prepare
# python dataPreparer.py --featuresFile datasets/4_noduplicate_features.csv --action randomDownsample --outputFile datasets/5_downsampled_100_no_other.csv --keptCells 100
python dataPreparer.py --featuresFile datasets/5_downsampled_100_no_other.csv --action reduce --feature_reductor SelectFromModel --nr_reduced_features 120 --outputFile datasets/6_select_120_fromModel_no_other.csv

# ./activeLearner.sh
