import argparse
import random
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import (LabelEncoder, MaxAbsScaler, MinMaxScaler,
                                   Normalizer, PowerTransformer,
                                   QuantileTransformer, RobustScaler,
                                   StandardScaler, minmax_scale)
from sklearn.tree import DecisionTreeClassifier

from experiment_setup_lib import train_and_evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--random_data', action='store_true')
parser.add_argument('--dataset_path')
parser.add_argument('--classifier',
                    required=True,
                    help="Supported types: RF, DTree, NB, SVM, Linear, Norm")
parser.add_argument('--cores', type=int, default=-1)
parser.add_argument('--output_dir', default='tmp/')
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--train_fraction', type=float, default=0.5)

config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()

np.random.seed(config.random_seed)
random.seed(config.random_seed)

# Read in dataset into pandas dataframe
df = pd.read_csv(config.dataset_path, index_col="id")

# shuffle df
df = df.sample(frac=1, random_state=config.random_seed).reset_index(drop=True)

# feature normalization
print(df)

scaler = RobustScaler()
feature_col_names = list(filter(lambda c: c != 'CLASS', df.columns))
df[feature_col_names] = scaler.fit_transform(df[feature_col_names])

print(df)

# feature selection

# train/test split
X_train = df.sample(frac=config.train_fraction,
                    random_state=config.random_seed)
X_test = df.drop(X_train.index)

Y_train = X_train.pop('CLASS')
Y_test = X_test.pop('CLASS')

label_encoder = LabelEncoder()
label_encoder.fit(Y_train)
label_encoder.fit(Y_test)

Y_train = label_encoder.transform(Y_train)
Y_test = label_encoder.transform(Y_test)

#  print(X_train)
#  print(Y_train)
#  print(X_test)
#  print(Y_test)

# train baseline models

if config.classifier == "RF":
    clf = RandomForestClassifier(random_state=config.random_seed,
                                 n_jobs=config.cores,
                                 verbose=100)
if config.classifier == "SVM":
    clf = svm.LinearSVC(random_state=config.random_seed, verbose=100)
elif config.classifier == "DTree":
    clf = DecisionTreeClassifier(random_state=config.random_seed)
elif config.classifier == "NB":
    clf = MultinomialNB()

train_and_evaluate(clf, X_train, Y_train, X_test, Y_test, config,
                   label_encoder)
