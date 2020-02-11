import argparse
import contextlib
import io
import os
import random
import sys

import numpy as np
from sklearn.model_selection import train_test_split

from cluster_strategies import DummyClusterStrategy, RandomClusterStrategy, MostUncertainClusterStrategy, RoundRobinClusterStrategy
from dataStorage import DataStorage
from experiment_setup_lib import (Logger,
                                  classification_report_and_confusion_matrix,
                                  load_and_prepare_X_and_Y, standard_config,
                                  store_pickle, store_result)
from sampling_strategies import (BoundaryPairSampler, CommitteeSampler,
                                 RandomSampler, UncertaintySampler)

config = standard_config([
    (['--sampling'], {
        'required': True,
        'help': "Possible values: uncertainty, random, committe, boundary"
    }),
    (['--cluster'], {
        'default': 'dummy',
        'help': "Possible values: dummy, random, mostUncertain, roundRobin"
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
    (['--with_recommendation'], {
        'action': 'store_true'
    }),
    (['--with_cluster_recommendation'], {
        'action': 'store_true'
    }),
    (['--with_snuba_lite'], {
        'action': 'store_true'
    }),
    (['--plot'], {
        'action': 'store_true'
    }),
])

dataStorage = DataStorage(config)
dataStorage.load_csv(config.dataset_path)

if config.sampling == 'random':
    active_learner = RandomSampler(config)
elif config.sampling == 'boundary':
    active_learner = BoundaryPairSampler(config)
elif config.sampling == 'uncertainty':
    active_learner = UncertaintySampler(config)
    active_learner.set_uncertainty_strategy('least_confident')
elif config.sampling == 'uncertainty_max_margin':
    active_learner = UncertaintySampler(config)
    active_learner.set_uncertainty_strategy('max_margin')
elif config.sampling == 'uncertainty_entropy':
    active_learner = UncertaintySampler(config)
    active_learner.set_uncertainty_strategy('entropy')
elif config.sampling == 'committee':
    active_learner = CommitteeSampler(config)
else:
    print("No Active Learning Strategy specified")
    exit(-4)

if config.cluster == 'dummy':
    cluster_strategy = DummyClusterStrategy()
elif config.cluster == 'random':
    cluster_strategy = RandomClusterStrategy()
elif config.cluster == "MostUncertain":
    cluster_strategy = MostUncertainClusterStrategy()
elif config.cluster == 'RoundRobin':
    cluster_strategy = RoundRobinClusterStrategy()

filename = config.sampling + '_' + str(config.start_set_size) + '_' + str(
    config.nr_queries_per_iteration)

store_result(filename + ".txt", "", config)
with Logger(config.output_dir + '/' + filename + ".txt", "w"):
    active_learner.set_data_storage(dataStorage)
    cluster_strategy.set_data_storage(dataStorage)
    active_learner.set_cluster_strategy(cluster_strategy)

    trained_active_clf_list, metrics_per_al_cycle = active_learner.learn()

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
                                           dataStorage.X_test,
                                           dataStorage.Y_test,
                                           config,
                                           dataStorage.label_encoder,
                                           output_dict=False,
                                           store=False,
                                           training_times="")
