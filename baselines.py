import argparse
import random
import sys
from sklearn.model_selection import train_test_split
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
from sklearn.feature_selection import SelectKBest, chi2
from experiment_setup_lib import train_and_evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--random_data', action='store_true')
parser.add_argument('--dataset_path')
parser.add_argument('--classifier',
                    required=True,
                    help="Supported types: RF, DTree, NB, SVM, Linear")
parser.add_argument('--cores', type=int, default=-1)
parser.add_argument('--output_dir', default='tmp/')
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--test_fraction', type=float, default=0.5)

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

# create numpy data
Y = df.pop('CLASS').to_numpy()

label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

X = df.to_numpy()

# feature normalization
scaler = RobustScaler()
X = scaler.fit_transform(X)

# scale again to [0,1]
#  scaler = MinMaxScaler()
#  X = scaler.fit_transform(X)

# feature selection
#  selector = SelectKBest(chi2, k=200)
#  X = selector.fit_transform(X, Y)

# train/test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=config.test_fraction, random_state=config.random_seed)

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
