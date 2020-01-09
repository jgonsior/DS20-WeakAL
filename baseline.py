from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
from experiment_setup_lib import train_and_evaluate, load_and_prepare_X_and_Y, standard_config

config = standard_config()

X, Y, label_encoder = load_and_prepare_X_and_Y(config)

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
