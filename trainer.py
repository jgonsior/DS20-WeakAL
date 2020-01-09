import argparse
import os
import random
import sys
import contextlib
import numpy as np
import io
from active_learning_strategies import (BoundaryPairSampler, CommitteeSampler,
                                        RandomSampler, UncertaintySampler)
from experiment_setup_lib import (load_and_prepare_X_and_Y, standard_config,
                                  store_pickle, store_result)
from sklearn.model_selection import train_test_split

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
    (['--start_set_size'], {
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
    test_size=1 - config.start_set_size,
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

f = io.StringIO()

with contextlib.redirect_stdout(f):
    active_learner.set_data(X_train_labeled, Y_train_labeled,
                            X_train_unlabeled, Y_train_unlabeled, X_test,
                            Y_test, label_encoder)
    trained_active_clf_list, metrics_per_al_cycle = active_learner.learn()

log = f.getvalue()
#  log = "hui"
#  trained_active_clf_list = ["ui"]
#  metrics_per_al_cycle = "oha"
filename = config.strategy + '_' + str(config.start_set_size) + '_' + str(
    config.nr_queries_per_iteration)

store_result(filename + ".txt", log, config)

# save output
store_pickle(filename + '.pickle', metrics_per_al_cycle, config)
