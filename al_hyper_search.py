import argparse
import contextlib
import io
import os
import random
import sys

import numpy as np
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     RandomizedSearchCV, train_test_split)

from cluster_strategies import (DummyClusterStrategy,
                                MostUncertainClusterStrategy,
                                RandomClusterStrategy,
                                RoundRobinClusterStrategy)
from dataStorage import DataStorage
from experiment_setup_lib import (Logger,
                                  classification_report_and_confusion_matrix,
                                  load_and_prepare_X_and_Y, standard_config,
                                  store_pickle, store_result)
from sampling_strategies import (BoundaryPairSampler, CommitteeSampler,
                                 RandomSampler, UncertaintySampler)

standard_config = standard_config()

param_grid = {
    "dataset_path": [standard_config.dataset_path],
    "classifier": [standard_config.classifier],
    "cores": [standard_config.cores],
    "output_dir ": [standard_config.output_dir],
    "random_seed": [standard_config.random_seed],
    "test_fraction": [standard_config.test_fraction],
    "sampling": ['random', 'comittee', 'boundary'],
    "cluster": [
        'dummy', 'random', 'RoundRobin', 'MostUncertain_lc',
        'MostUncertain_max_margin', 'MostUncertain_entropy'
    ],
    "nr_learning_iterations": [1000000000],
    "nr_queries_per_iteration":
    randint(1, 1000),
    "start_set_size":
    uniform(0, 0.5),
    "minimum_test_accuracy_before_recommendations":
    uniform(0.5, 1),
    "uncertainty_recommendation_certainty_threshold":
    uniform(0.5, 1),
    "uncertainty_recommendation_ratio": [1 / 10, 1 / 100, 1 / 1000, 1 / 10000],
    "snuba_lite_minimum_heuristic_accuracy":
    uniform(0.5, 1),
    "cluster_recommendation_minimum_cluster_unity_size":
    uniform(0.5, 1),
    "cluster_recommendation_ratio_labeled_unlabeled":
    uniform(0.5, 1),
    "with_uncertainty_recommendation": [True, False],
    "with_cluster_recommendation": [True, False],
    "with_snuba_lite": [False],
    "plot": [True],
}

# create estimater object


class Estimator(BaseEstimator):
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b

    def fit(self, X, y, **kwargs):
        print("got called")

    def score(self, X, y):
        return 1


param_grid = {'a': [1, 2], 'b': [True, False]}
active_learner = Estimator()
print(active_learner.get_params().keys())
grid = RandomizedSearchCV(active_learner,
                          param_grid,
                          verbose=9999999999999999999999999999999999)

dataStorage = DataStorage(standard_config)
dataStorage.load_csv(standard_config.dataset_path)

grid.fit(dataStorage.X, dataStorage.Y)
exit(-2)

if config.sampling == 'random':
    active_learner = RandomSampler(config)
elif config.sampling == 'boundary':
    active_learner = BoundaryPairSampler(config)
elif config.sampling == 'uncertainty_lc':
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
elif config.cluster == "MostUncertain_lc":
    cluster_strategy = MostUncertainClusterStrategy()
    cluster_strategy.set_uncertainty_strategy('least_confident')
elif config.cluster == "MostUncertain_max_margin":
    cluster_strategy = MostUncertainClusterStrategy()
    cluster_strategy.set_uncertainty_strategy('max_margin')
elif config.cluster == "MostUncertain_entropy":
    cluster_strategy = MostUncertainClusterStrategy()
    cluster_strategy.set_uncertainty_strategy('entropy')
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
