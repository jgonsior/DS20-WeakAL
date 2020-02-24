import argparse
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
                                     RandomizedSearchCV, train_test_split)
from sklearn.preprocessing import LabelEncoder

from cluster_strategies import (DummyClusterStrategy,
                                MostUncertainClusterStrategy,
                                RandomClusterStrategy,
                                RoundRobinClusterStrategy)
from dataStorage import DataStorage
from experiment_setup_lib import (ExperimentResult, Logger,
                                  classification_report_and_confusion_matrix,
                                  get_db, load_and_prepare_X_and_Y,
                                  standard_config, store_pickle, store_result)
from sampling_strategies import (BoundaryPairSampler, CommitteeSampler,
                                 RandomSampler, UncertaintySampler)

standard_config = standard_config([
    (['--nr_learning_iterations'], {
        'type': int,
        'default': 2000000
    }),
    (['--cv'], {
        'type': int,
        'default': 3
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
        'default': 3
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
])

logging_file_name = standard_config.output_dir + "/" + str(
    datetime.datetime.now()) + "al_hyper_search.txt"

logging.basicConfig(
    filename=logging_file_name,
    filemode='a',
    level=logging.INFO,
    format="[%(process)d] [%(asctime)s] %(levelname)s: %(message)s")

param_distribution = {}

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
    np.linspace(1, 2000, num=51).astype(int),
    "start_set_size":
    np.linspace(0.01, 0.2, num=10).astype(float),
    "stopping_criteria_uncertainty":
    np.linspace(0, 1, num=101).astype(float),
    "stopping_criteria_std":
    np.linspace(0, 1, num=101).astype(float),
    "stopping_criteria_acc":
    np.linspace(0, 1, num=101).astype(float),
    "allow_recommendations_after_stop": [True, False],

    #uncertainty_recommendation_grid = {
    "uncertainty_recommendation_certainty_threshold":
    np.linspace(0.5, 1, num=51).astype(float),
    "uncertainty_recommendation_ratio": [1 / 10, 1 / 100, 1 / 1000, 1 / 10000],

    #snuba_lite_grid = {
    "snuba_lite_minimum_heuristic_accuracy":
    np.linspace(0.5, 1, num=51).astype(float),

    #cluster_recommendation_grid = {
    "cluster_recommendation_minimum_cluster_unity_size":
    np.linspace(0.5, 1, num=51).astype(float),
    "cluster_recommendation_ratio_labeled_unlabeled":
    np.linspace(0.5, 1, num=51).astype(float),
    "with_uncertainty_recommendation": [True, False],
    "with_cluster_recommendation": [True, False],
    "with_snuba_lite": [False],
    "minimum_test_accuracy_before_recommendations":
    np.linspace(0.5, 1, num=51).astype(float),
}

db = get_db(db_name_or_type=standard_config.db)


class Estimator(BaseEstimator):
    def __init__(self,
                 label_encoder_classes=None,
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
                 allow_recommendations_after_stop=None):
        self.label_encoder_classes = label_encoder_classes
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

    def fit(self, X_train, Y_train, **kwargs):
        self.len_train_data = len(Y_train)
        label_encoder = LabelEncoder()
        label_encoder.fit(self.label_encoder_classes)
        self.dataset_storage = DataStorage(self.random_seed)
        self.dataset_storage.set_training_data(X_train, Y_train, label_encoder,
                                               self.test_fraction,
                                               self.start_set_size)

        if self.cluster == 'dummy':
            cluster_strategy = DummyClusterStrategy()
        elif self.cluster == 'random':
            cluster_strategy = RandomClusterStrategy()
        elif self.cluster == "MostUncertain_lc":
            cluster_strategy = MostUncertainClusterStrategy()
            cluster_strategy.set_uncertainty_strategy('least_confident')
        elif self.cluster == "MostUncertain_max_margin":
            cluster_strategy = MostUncertainClusterStrategy()
            cluster_strategy.set_uncertainty_strategy('max_margin')
        elif self.cluster == "MostUncertain_entropy":
            cluster_strategy = MostUncertainClusterStrategy()
            cluster_strategy.set_uncertainty_strategy('entropy')
        elif self.cluster == 'RoundRobin':
            cluster_strategy = RoundRobinClusterStrategy()

        cluster_strategy.set_data_storage(self.dataset_storage)

        active_learner_params = {
            'dataset_storage': self.dataset_storage,
            'cluster_strategy': cluster_strategy,
            'cores': self.cores,
            'random_seed': self.random_seed,
            'nr_learning_iterations': self.nr_learning_iterations,
            'nr_queries_per_iteration': self.nr_queries_per_iteration,
            'with_test': False,
        }

        if self.sampling == 'random':
            active_learner = RandomSampler(**active_learner_params)
        elif self.sampling == 'boundary':
            active_learner = BoundaryPairSampler(**active_learner_params)
        elif self.sampling == 'uncertainty_lc':
            active_learner = UncertaintySampler(**active_learner_params)
            active_learner.set_uncertainty_strategy('least_confident')
        elif self.sampling == 'uncertainty_max_margin':
            active_learner = UncertaintySampler(**active_learner_params)
            active_learner.set_uncertainty_strategy('max_margin')
        elif self.sampling == 'uncertainty_entropy':
            active_learner = UncertaintySampler(**active_learner_params)
            active_learner.set_uncertainty_strategy('entropy')
        #  elif self.sampling == 'committee':
        #  active_learner = CommitteeSampler(self.random_seed, self.cores, self.nr_learning_iterations)
        else:
            logging.error("No Active Learning Strategy specified")

        start = timer()
        trained_active_clf_list, metrics_per_al_cycle = active_learner.learn(
            minimum_test_accuracy_before_recommendations=self.
            minimum_test_accuracy_before_recommendations,
            with_cluster_recommendation=self.with_cluster_recommendation,
            with_uncertainty_recommendation=self.
            with_uncertainty_recommendation,
            with_snuba_lite=self.with_snuba_lite,
            cluster_recommendation_minimum_cluster_unity_size=self.
            cluster_recommendation_minimum_cluster_unity_size,
            cluster_recommendation_minimum_ratio_labeled_unlabeled=self.
            cluster_recommendation_ratio_labeled_unlabeled,
            uncertainty_recommendation_certainty_threshold=self.
            uncertainty_recommendation_certainty_threshold,
            uncertainty_recommendation_ratio=self.
            uncertainty_recommendation_ratio,
            snuba_lite_minimum_heuristic_accuracy=self.
            snuba_lite_minimum_heuristic_accuracy,
            stopping_criteria_uncertainty=self.stopping_criteria_uncertainty,
            stopping_criteria_acc=self.stopping_criteria_acc,
            stopping_criteria_std=self.stopping_criteria_std,
            allow_recommendations_after_stop=self.
            allow_recommendations_after_stop)
        end = timer()

        self.trained_active_clf_list = trained_active_clf_list
        self.fit_time = end - start
        self.metrics_per_al_cycle = metrics_per_al_cycle
        self.active_learner = active_learner

    def score(self, X_test, Y_test):
        # display quick results
        self.amount_of_user_asked_queries = self.active_learner.amount_of_user_asked_queries

        classification_report_and_confusion_matrix_test = classification_report_and_confusion_matrix(
            self.trained_active_clf_list[0], X_test, Y_test,
            self.dataset_storage.label_encoder)
        classification_report_and_confusion_matrix_train = classification_report_and_confusion_matrix(
            self.trained_active_clf_list[0],
            self.dataset_storage.X_train_labeled,
            self.dataset_storage.Y_train_labeled,
            self.dataset_storage.label_encoder)

        # normalize by start_set_size
        percentage_user_asked_queries = 1 - self.amount_of_user_asked_queries / self.len_train_data
        test_acc = classification_report_and_confusion_matrix_test[0][
            'accuracy']

        # score is harmonic mean
        score = 2 * percentage_user_asked_queries * test_acc / (
            percentage_user_asked_queries + test_acc)

        # calculate based on params a unique id which should be the same across all similar cross validation splits
        unique_params = ""

        for k in param_distribution.keys():
            unique_params += str(vars(self)[k])

        param_list_id = hashlib.md5(unique_params.encode('utf-8')).hexdigest()

        experiment_result = ExperimentResult(
            **self.get_params(),
            amount_of_user_asked_queries=self.amount_of_user_asked_queries,
            metrics_per_al_cycle=dumps(self.metrics_per_al_cycle,
                                       allow_nan=True),
            fit_time=str(self.fit_time),
            confusion_matrix_test=dumps(
                classification_report_and_confusion_matrix_test[1],
                allow_nan=True),
            confusion_matrix_train=dumps(
                classification_report_and_confusion_matrix_train[1],
                allow_nan=True),
            classification_report_test=dumps(
                classification_report_and_confusion_matrix_test[0],
                allow_nan=True),
            classification_report_train=dumps(
                classification_report_and_confusion_matrix_train[0],
                allow_nan=True),
            acc_train=classification_report_and_confusion_matrix_train[0]
            ['accuracy'],
            acc_test=classification_report_and_confusion_matrix_test[0]
            ['accuracy'],
            fit_score=score,
            param_list_id=param_list_id)
        experiment_result.save()

        return score


active_learner = Estimator()

X, Y, label_encoder = load_and_prepare_X_and_Y(standard_config.dataset_path)

param_distribution['label_encoder_classes'] = [label_encoder.classes_]

#  grid = RandomizedSearchCV(active_learner,
#  param_distribution,
#  n_iter=standard_config.nr_random_runs,
#  cv=standard_config .cv,
#  verbose=9999999999999999999999999999999999)
#  n_jobs=multiprocessing.cpu_count())

grid = EvolutionaryAlgorithmSearchCV(
    estimator=active_learner,
    params=param_distribution,
    verbose=True,
    cv=standard_config.cv,
    population_size=standard_config.population_size,
    gene_mutation_prob=standard_config.gene_mutation_prob,
    tournament_size=standard_config.tournament_size,
    generations_number=standard_config.generations_number,
    n_jobs=multiprocessing.cpu_count())

#  search = grid.fit(X, Y)
grid.fit(X, Y)

print(grid.best_params_)
print(grid.best_score_)
print(
    pd.DataFrame(grid.cv_results_).sort_values("mean_test_score",
                                               ascending=False).head())
