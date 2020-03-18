#!/bin/bash
python baseline.py --classifier RF --dataset_path ~/coding/datasets/dwtc/aft.csv --output_dir results/baselines/rf
python baseline.py --classifier DTree --dataset_path ~/coding/datasets/dwtc/aft.csv --output_dir results/baselines/dtree
python baseline.py --classifier NB --dataset_path ~/coding/datasets/dwtc/aft.csv --output_dir results/baselines/nb
python baseline.py --classifier SVM --dataset_path ~/coding/datasets/dwtc/aft.csv --output_dir results/baselines/svm
