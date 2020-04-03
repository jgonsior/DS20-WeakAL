#!/bin/bash
python baseline.py --CLASSIFIER RF --dataset_path ~/coding/datasets/dwtc/aft.csv --output_dir results/baselines/rf
python baseline.py --CLASSIFIER DTree --dataset_path ~/coding/datasets/dwtc/aft.csv --output_dir results/baselines/dtree
python baseline.py --CLASSIFIER NB --dataset_path ~/coding/datasets/dwtc/aft.csv --output_dir results/baselines/nb
python baseline.py --CLASSIFIER SVM --dataset_path ~/coding/datasets/dwtc/aft.csv --output_dir results/baselines/svm
