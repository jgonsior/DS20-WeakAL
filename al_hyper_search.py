import argparse
import gc
import contextlib
import datetime
import hashlib
import io
import logging
import multiprocessing
import os
import random
import sys
from itertools import chain, combinations
from timeit import default_timer as timer

import numpy as np
#  import np.random.distributions as dists
import numpy.random
import pandas as pd
import peewee
import scipy
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from json_tricks import dumps
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     RandomizedSearchCV, ShuffleSplit,
                                     train_test_split)
from sklearn.preprocessing import LabelEncoder

from cluster_strategies import (DummyClusterStrategy,
                                MostUncertainClusterStrategy,
                                RandomClusterStrategy,
                                RoundRobinClusterStrategy)
from dataStorage import DataStorage
from experiment_setup_lib import (ExperimentResult, Logger,
                                  classification_report_and_confusion_matrix,
                                  get_db, load_and_prepare_X_and_Y,
                                  divide_data, standard_config, store_pickle,
                                  store_result, get_dataset, init_logging)
from al_cycle_wrapper import train_and_eval_dataset
from sampling_strategies import (BoundaryPairSampler, CommitteeSampler,
                                 RandomSampler, UncertaintySampler)

standard_config = standard_config([
    (['--nr_learning_iterations'], {
        'type': int,
        'default': 100
    }),
    (['--cv'], {
        'type': int,
        'default': 3
    }),
    (['--n_jobs'], {
        'type': int,
        'default': multiprocessing.cpu_count()
    }),
    (['--nr_random_runs'], {
        'type': int,
        'default': 200000
    }),
    (['--population_size'], {
        'type': int,
        'default': 100
    }),
    (['--tournament_size'], {
        'type': int,
        'default': 100
    }),
    (['--generations_number'], {
        'type': int,
        'default': 1000
    }),
    (['--gene_mutation_prob'], {
        'type': float,
        'default': 0.3
    }),
    (['--db'], {
        'default': 'sqlite'
    }),
    (['--hyper_search_type'], {
        'default': 'random'
    }),
])

init_logging(standard_config.output_dir, level=logging.INFO)
#  logging_file_name = standard_config.output_dir + "/" + str(
#  datetime.datetime.now()) + "al_hyper_search.txt"

#  logging.basicConfig(
#  #  filename=logging_file_name,
#  #  filemode='a',
#  level=logging.INFO,
#  format="[%(process)d] [%(asctime)s] %(levelname)s: %(message)s")

param_size = 50
#  param_size = 2

param_distribution = {
    "dataset_path": [standard_config.dataset_path],
    "classifier": [standard_config.classifier],
    "cores": [standard_config.cores],
    "output_dir": [standard_config.output_dir],
    "random_seed": [standard_config.random_seed],
    "test_fraction": [standard_config.test_fraction],
    "sampling": [
        'random',
        'uncertainty_lc',
        'uncertainty_max_margin',
        'uncertainty_entropy',
    ],
    "cluster": [
        'dummy', 'random', 'MostUncertain_lc', 'MostUncertain_max_margin',
        'MostUncertain_entropy'
        #  'dummy',
    ],
    "nr_learning_iterations": [standard_config.nr_learning_iterations],
    #  "nr_learning_iterations": [1],
    "nr_queries_per_iteration":
    np.linspace(1, 150, num=param_size + 1).astype(int),
    "start_set_size":
    np.linspace(0.01, 0.2, num=10).astype(float),
    "stopping_criteria_uncertainty":
    np.linspace(0, 1, num=param_size * 2 + 1).astype(float),
    "stopping_criteria_std":
    np.linspace(0, 1, num=param_size * 2 + 1).astype(float),
    "stopping_criteria_acc":
    np.linspace(0, 1, num=param_size * 2 + 1).astype(float),
    "allow_recommendations_after_stop": [True, False],

    #uncertainty_recommendation_grid = {
    "uncertainty_recommendation_certainty_threshold":
    np.linspace(0.5, 1, num=param_size + 1).astype(float),
    "uncertainty_recommendation_ratio": [1 / 10, 1 / 100, 1 / 1000, 1 / 10000],

    #snuba_lite_grid = {
    "snuba_lite_minimum_heuristic_accuracy":
    np.linspace(0.5, 1, num=param_size + 1).astype(float),

    #cluster_recommendation_grid = {
    "cluster_recommendation_minimum_cluster_unity_size":
    np.linspace(0.5, 1, num=param_size + 1).astype(float),
    "cluster_recommendation_ratio_labeled_unlabeled":
    np.linspace(0.5, 1, num=param_size + 1).astype(float),
    "with_uncertainty_recommendation": [True, False],
    "with_cluster_recommendation": [True, False],
    "with_snuba_lite": [False],
    "minimum_test_accuracy_before_recommendations":
    np.linspace(0.5, 1, num=param_size + 1).astype(float),
    "db_name_or_type": [standard_config.db],
}


class Estimator(BaseEstimator):
    def __init__(self,
                 dataset_path=None,
                 classifier=None,
                 cores=None,
                 output_dir=None,
                 random_seed=None,
                 test_fraction=None,
                 sampling=None,
                 cluster=None,
                 nr_learning_iterations=None,
                 nr_queries_per_iteration=None,
                 start_set_size=None,
                 minimum_test_accuracy_before_recommendations=None,
                 uncertainty_recommendation_certainty_threshold=None,
                 uncertainty_recommendation_ratio=None,
                 snuba_lite_minimum_heuristic_accuracy=None,
                 cluster_recommendation_minimum_cluster_unity_size=None,
                 cluster_recommendation_ratio_labeled_unlabeled=None,
                 with_uncertainty_recommendation=None,
                 with_cluster_recommendation=None,
                 with_snuba_lite=None,
                 plot=None,
                 stopping_criteria_uncertainty=None,
                 stopping_criteria_std=None,
                 stopping_criteria_acc=None,
                 allow_recommendations_after_stop=None,
                 db_name_or_type=None):
        self.dataset_path = dataset_path
        self.classifier = classifier
        self.cores = cores
        self.output_dir = output_dir
        self.random_seed = random_seed
        self.test_fraction = test_fraction
        self.sampling = sampling
        self.cluster = cluster
        self.nr_learning_iterations = nr_learning_iterations
        self.nr_queries_per_iteration = nr_queries_per_iteration
        self.start_set_size = start_set_size
        self.minimum_test_accuracy_before_recommendations = minimum_test_accuracy_before_recommendations
        self.uncertainty_recommendation_certainty_threshold = uncertainty_recommendation_certainty_threshold
        self.uncertainty_recommendation_ratio = uncertainty_recommendation_ratio
        self.snuba_lite_minimum_heuristic_accuracy = snuba_lite_minimum_heuristic_accuracy
        self.cluster_recommendation_minimum_cluster_unity_size = cluster_recommendation_minimum_cluster_unity_size
        self.cluster_recommendation_ratio_labeled_unlabeled = cluster_recommendation_ratio_labeled_unlabeled
        self.with_uncertainty_recommendation = with_uncertainty_recommendation
        self.with_cluster_recommendation = with_cluster_recommendation
        self.with_snuba_lite = with_snuba_lite
        self.plot = plot
        self.stopping_criteria_acc = stopping_criteria_acc
        self.stopping_criteria_std = stopping_criteria_std
        self.stopping_criteria_uncertainty = stopping_criteria_uncertainty
        self.allow_recommendations_after_stop = allow_recommendations_after_stop

        self.db_name_or_type = db_name_or_type

    def fit(self, dataset_names, Y_not_used, **kwargs):

        self.scores = []
        for dataset_name in dataset_names:
            gc.collect()

            X_train, X_test, Y_train, Y_test, label_encoder_classes = get_dataset(
                standard_config.dataset_path, dataset_name)

            self.scores.append(
                train_and_eval_dataset(dataset_name, X_train, X_test, Y_train,
                                       Y_test, label_encoder_classes, self,
                                       param_distribution))

            X_train = X_test = Y_train = Y_test = label_encoder_classes = None
            gc.collect()

    def score(self, dataset_names, Y_not_used):
        for dataset_name in dataset_names:
            gc.collect()

            X_train, X_test, Y_train, Y_test, label_encoder_classes = get_dataset(
                standard_config.dataset_path, dataset_name)

            self.scores.append(
                train_and_eval_dataset(dataset_name, X_train, X_test, Y_train,
                                       Y_test, label_encoder_classes, self,
                                       param_distribution))
            X_train = X_test = Y_train = Y_test = label_encoder_classes = None
            gc.collect()
        return sum(self.scores) / len(self.scores)


active_learner = Estimator()

#  X = ['dwtc', 'ibn_sina', 'hiva', 'orange', 'sylva', 'zebra']
X = ['forest_covtype', 'forest_covtype']
Y = [None] * len(X)

if standard_config.hyper_search_type == 'random':
    grid = RandomizedSearchCV(active_learner,
                              param_distribution,
                              n_iter=standard_config.nr_random_runs,
                              cv=2,
                              verbose=9999999999999999999999999999999999,
                              n_jobs=multiprocessing.cpu_count())
    grid = grid.fit(X, Y)
elif standard_config.hyper_search_type == 'evo':
    grid = EvolutionaryAlgorithmSearchCV(
        estimator=active_learner,
        params=param_distribution,
        verbose=True,
        cv=ShuffleSplit(test_size=0.20, n_splits=1,
                        random_state=0),  # fake CV=1 split
        population_size=standard_config.population_size,
        gene_mutation_prob=standard_config.gene_mutation_prob,
        tournament_size=standard_config.tournament_size,
        generations_number=standard_config.generations_number,
        n_jobs=standard_config.n_jobs)
    grid.fit(X, Y)

print(grid.best_params_)
print(grid.best_score_)
print(
    pd.DataFrame(grid.cv_results_).sort_values("mean_test_score",
                                               ascending=False).head())
