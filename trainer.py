import argparse
from sklearn.model_selection import train_test_split
import os
import random
import sys

import numpy as np
from experiment_setup_lib import load_and_prepare_X_and_Y, standard_config
from active_learning_strategies import BoundaryPairSampler, CommitteeSampler, RandomSampler, UncertaintySampler

config = standard_config([
    (['--strategy'], {
        'required': True,
        'help': "Possible values: uncertainty, random, committe, boundary"
    }),
    (['--nr_learning_iterations'], {
        'type': int,
        'default': 15
    }),
    (['--nr_queries_per_iteration'], {
        'type': int,
        'default': 150
    }),
    (['--start_size'], {
        'type': float,
        'default': 0.1
    }),
    (['--plot'], {
        'action': 'store_true'
    }),
])

X, Y, label_encoder = load_and_prepare_X_and_Y(config)

# split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=config.test_fraction, random_state=config.random_seed)

# split training data into labeled and unlabeled dataset
X_train_labeled, X_train_unlabeled, Y_train_labeled, Y_train_unlabeled = train_test_split(
    X_train,
    Y_train,
    test_size=config.start_size,
    random_state=config.random_seed)

if config.strategy == 'random':
    active_learner = RandomSampler(config)
elif config.strategy == 'boundary':
    active_learner = BoundaryPairSampler(config)
elif config.strategy == 'uncertainty':
    active_learner = UncertaintySampler(config)
    active_learner.set_uncertainty_strategy('least_confident')
elif config.strategy == 'uncertainty_max_margin':
    active_learner = UncertaintySampler(config)
    active_learner.set_uncertainty_strategy('max_margin')
elif config.strategy == 'uncertainty_entropy':
    active_learner = UncertaintySampler(config)
    active_learner.set_uncertainty_strategy('entropy')
elif config.strategy == 'committee':
    active_learner = CommitteeSampler(config)
else:
    print("No Active Learning Strategy specified")
    exit(-4)

active_learner.set_data(X_train_labeled, Y_train_labeled, X_train_unlabeled,
                        Y_train_unlabeled, X_test, Y_test, label_encoder)
trained_active_clf, log = active_learner.learn()

Y_train_unlabeled_predicted = trained_active_clf.predict(X_train_unlabeled)
Y_test_predicted = trained_active_clf.predict(X_test)
