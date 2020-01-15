import argparse
import contextlib
import io
import os
import random
import sys

import numpy as np

from active_learning_strategies import (BoundaryPairSampler, CommitteeSampler,
                                        RandomSampler, UncertaintySampler)
from experiment_setup_lib import (Logger,
                                  classification_report_and_confusion_matrix,
                                  load_and_prepare_X_and_Y, standard_config,
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
    X, Y, test_size=config.test_fraction)

# split training data into labeled and unlabeled dataset
X_train_labeled, X_train_unlabeled, Y_train_labeled, Y_train_unlabeled = train_test_split(
    X_train, Y_train, test_size=1 - config.start_set_size)

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

filename = config.strategy + '_' + str(config.start_set_size) + '_' + str(
    config.nr_queries_per_iteration)

store_result(filename + ".txt", "", config)
with Logger(config.output_dir + '/' + filename + ".txt", "w"):
    active_learner.set_data(X_train_labeled, Y_train_labeled,
                            X_train_unlabeled, Y_train_unlabeled, X_test,
                            Y_test, label_encoder)
    trained_active_clf_list, metrics_per_al_cycle = active_learner.learn()

#  log = "hui"
#  trained_active_clf_list = ["ui"]
#  metrics_per_al_cycle = "oha"

# save output
store_pickle(filename + '.pickle', metrics_per_al_cycle, config)

# display quick results

amount_of_user_asked_queries = 0

for i, amount_of_queries in enumerate(metrics_per_al_cycle['query_length']):
    if metrics_per_al_cycle['recommendation'][i] == "A":
        amount_of_user_asked_queries += amount_of_queries

print(
    "User were asked to label {} queries".format(amount_of_user_asked_queries))

classification_report_and_confusion_matrix(trained_active_clf_list[0],
                                           X_test,
                                           Y_test,
                                           config,
                                           label_encoder,
                                           output_dict=False,
                                           store=False,
                                           training_times="")
