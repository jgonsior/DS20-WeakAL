import argparse
import random
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from experiment_setup_lib import train_and_evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--random_data', action='store_true')
parser.add_argument('--dataset_path')
parser.add_argument('--classifier',
                    required=True,
                    help="Supported types: RF, DTree, LR, SVM")
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

# 1. Read in dataset into pandas dataframe
df = pd.read_csv(config.dataset_path, index_col="id")
#  print(df)

# 2. train/test split
X_train = df.sample(frac=config.train_fraction,
                    random_state=config.random_seed)
X_test = df.drop(X_train.index)
Y_train = X_train.pop('CLASS')
Y_test = X_test.pop('CLASS')

#  print(X_train)
#  print(Y_train)
#  print(X_test)
#  print(Y_test)

# 3. feature selection

# 4. train baseline models

if config.classifier == "RF":
    clf = RandomForestClassifier(random_state=config.random_seed,
                                 n_jobs=config.cores,
                                 verbose=100)

train_and_evaluate(clf, X_train, Y_train, X_test, Y_test, config)
# 5. print out stats
