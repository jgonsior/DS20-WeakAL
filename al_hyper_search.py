import argparse
import contextlib
import datetime
import gc
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

from al_cycle_wrapper import train_and_eval_dataset
from cluster_strategies import (DummyClusterStrategy,
                                MostUncertainClusterStrategy,
                                RandomClusterStrategy,
                                RoundRobinClusterStrategy)
from dataStorage import DataStorage
from experiment_setup_lib import (ExperimentResult,
                                  classification_report_and_confusion_matrix,
                                  divide_data, get_dataset, get_db,
                                  get_param_distribution,
                                  load_and_prepare_X_and_Y, log_it,
                                  standard_config)
from sampling_strategies import (BoundaryPairSampler, CommitteeSampler,
                                 RandomSampler, UncertaintySampler)

standard_config = standard_config([
    (['--nr_learning_iterations'], {
        'type': int,
        'default': 1000000
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
        'default': 100
    }),
    (['--gene_mutation_prob'], {
        'type': float,
        'default': 0.3
    }),
    (['--db_name_or_type'], {
        'default': 'sqlite'
    }),
    (['--hyper_search_type'], {
        'default': 'random'
    }),
])

param_distribution = get_param_distribution(**vars(standard_config))


class Estimator(BaseEstimator):
    def __init__(self,
                 datasets_path=None,
                 classifier=None,
                 cores=None,
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
        self.datasets_path = datasets_path
        self.classifier = classifier
        self.cores = cores
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
        #  if len(kwargs) == 0:
        #  for k in get_param_distribution().keys():
        #  setattr(self, k, None)
        #  else:
        #  for k, v in kwargs.items():
        #  setattr(self, k, v)

    def fit(self, dataset_names, Y_not_used, **kwargs):
        #  print("fit", dataset_names)
        self.scores = []

        for dataset_name in dataset_names:
            if dataset_name is None:
                continue
            #  gc.collect()

            X_train, X_test, Y_train, Y_test, label_encoder_classes = get_dataset(
                standard_config.datasets_path, dataset_name)

            self.scores.append(
                train_and_eval_dataset(dataset_name,
                                       X_train,
                                       X_test,
                                       Y_train,
                                       Y_test,
                                       label_encoder_classes,
                                       hyper_parameters=vars(self)))

            log_it(dataset_name + " done with " + str(self.scores[-1]))
            #  gc.collect()

    def score(self, dataset_names_should_be_none, Y_not_used):
        #  print("score", dataset_names_should_be_none)
        return sum(self.scores) / len(self.scores)


active_learner = Estimator()

if standard_config.nr_learning_iterations == 3:
    X = ['dwtc', 'ibn_sina']
else:
    X = [
        'dwtc',
        'ibn_sina',
        'hiva',
        'orange',
        'sylva',
        'forest_covtype',
        'zebra',
    ]

X.append(None)
Y = [None] * len(X)


# fake our own stupid cv=1 split
class NoCvCvSplit:
    def __init__(self, n_splits=3, **kwargs):
        self.n_splits = n_splits
        pass

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

    def split(self, dataset_names, y=None, groups=None, split_quota=None):
        # everything goes into train once
        train_idx = [i for i in range(0, len(dataset_names) - 1)]

        yield train_idx, [len(dataset_names) - 1]


if standard_config.hyper_search_type == 'random':
    grid = RandomizedSearchCV(
        active_learner,
        param_distribution,
        n_iter=standard_config.nr_random_runs,
        pre_dispatch=standard_config.n_jobs,
        return_train_score=False,
        cv=NoCvCvSplit(n_splits=1),
         verbose=9999999999999999999999999999999999,
        # verbose=0,
        n_jobs=standard_config.n_jobs)
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
