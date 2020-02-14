import argparse
import contextlib
import io
import os
import random
import sys
from itertools import chain, combinations

import numpy as np
import pandas as pd
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
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

param_distribution = {}

standard_param_distribution = {
    "dataset_path": [standard_config.dataset_path],
    "classifier": [standard_config.classifier],
    "cores": [standard_config.cores],
    "output_dir": [standard_config.output_dir],
    "random_seed": [standard_config.random_seed],
    "test_fraction": [standard_config.test_fraction],
    "sampling": [
        'random', 'uncertainty_lc', 'uncertainty_max_margin',
        'uncertainty_entropy', 'boundary'
    ],
    "cluster": [
        #  'dummy', 'random', 'MostUncertain_lc', 'MostUncertain_max_margin',
        #  'MostUncertain_entropy'
        'dummy',
    ],
    "nr_learning_iterations": [1000000000],
    "nr_queries_per_iteration":
    np.random.randint(1000, 2000, size=100),
    "start_set_size":
    np.random.uniform(0.01, 0.5, size=100),
    "with_uncertainty_recommendation": [False],
    "with_cluster_recommendation": [False],
    "with_snuba_lite": [False]
}

uncertainty_recommendation_grid = {
    "uncertainty_recommendation_certainty_threshold":
    np.random.uniform(0.5, 1, size=100),
    "uncertainty_recommendation_ratio": [1 / 10, 1 / 100, 1 / 1000, 1 / 10000]
}

snuba_lite_grid = {
    "snuba_lite_minimum_heuristic_accuracy": np.random.uniform(0.5,
                                                               1,
                                                               size=100)
}

cluster_recommendation_grid = {
    "cluster_recommendation_minimum_cluster_unity_size":
    np.random.uniform(0.5, 1, size=100),
    "cluster_recommendation_ratio_labeled_unlabeled":
    np.random.uniform(0.5, 1, size=100)
}


# generate all possible combinations of the three recommendations
def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))


param_distribution_list = []

for recommendation_param_distributions in powerset([
    ("with_uncertainty_recommendation", uncertainty_recommendation_grid),
    ("with_cluster_recommendation", cluster_recommendation_grid)
]):
    param_distribution = {**standard_param_distribution}
    for recommendation_param_distribution in recommendation_param_distributions:
        param_distribution = {
            **param_distribution,
            **recommendation_param_distribution[1]
        }
        param_distribution[recommendation_param_distribution[0]] = [True]
        param_distribution[
            "minimum_test_accuracy_before_recommendations"] = np.random.uniform(
                0.5, 1, size=100)
    param_distribution_list.append(param_distribution)

# create estimater object


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
                 plot=None):
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

        self.dataset_storage = DataStorage(random_seed)

    def fit(self, X, Y, **kwargs):
        self.dataset_storage.load_csv(self.dataset_path)
        self.dataset_storage.divide_data(self.test_fraction,
                                         self.start_set_size)

        if self.sampling == 'random':
            active_learner = RandomSampler(self.random_seed, self.cores,
                                           self.nr_learning_iterations,
                                           self.nr_queries_per_iteration)
        elif self.sampling == 'boundary':
            active_learner = BoundaryPairSampler(self.random_seed, self.cores,
                                                 self.nr_learning_iterations,
                                                 self.nr_queries_per_iteration)
        elif self.sampling == 'uncertainty_lc':
            active_learner = UncertaintySampler(self.random_seed, self.cores,
                                                self.nr_learning_iterations,
                                                self.nr_queries_per_iteration)
            active_learner.set_uncertainty_strategy('least_confident')
        elif self.sampling == 'uncertainty_max_margin':
            active_learner = UncertaintySampler(self.random_seed, self.cores,
                                                self.nr_learning_iterations,
                                                self.nr_queries_per_iteration)
            active_learner.set_uncertainty_strategy('max_margin')
        elif self.sampling == 'uncertainty_entropy':
            active_learner = UncertaintySampler(self.random_seed, self.cores,
                                                self.nr_learning_iterations,
                                                self.nr_queries_per_iteration)
            active_learner.set_uncertainty_strategy('entropy')
        #  elif self.sampling == 'committee':
        #  active_learner = CommitteeSampler(self.random_seed, self.cores, self.nr_learning_iterations)
        else:
            print("No Active Learning Strategy specified")

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

        filename = self.sampling + '_' + str(self.start_set_size) + '_' + str(
            self.nr_queries_per_iteration)

        store_result(filename + ".txt", "", self.output_dir)

        with Logger(self.output_dir + '/' + filename + ".txt", "w"):
            active_learner.set_data_storage(self.dataset_storage)
            cluster_strategy.set_data_storage(self.dataset_storage)
            active_learner.set_cluster_strategy(cluster_strategy)

            trained_active_clf_list, metrics_per_al_cycle = active_learner.learn(
                self.minimum_test_accuracy_before_recommendations,
                self.with_cluster_recommendation,
                self.with_uncertainty_recommendation, self.with_snuba_lite,
                self.cluster_recommendation_minimum_cluster_unity_size,
                self.cluster_recommendation_ratio_labeled_unlabeled,
                self.uncertainty_recommendation_certainty_threshold,
                self.uncertainty_recommendation_ratio,
                self.snuba_lite_minimum_heuristic_accuracy)

        # save output
        store_pickle(filename + '.pickle', metrics_per_al_cycle,
                     self.output_dir)

        # display quick results
        amount_of_user_asked_queries = 0

        for i, amount_of_queries in enumerate(
                metrics_per_al_cycle['query_length']):
            if metrics_per_al_cycle['recommendation'][i] == "A":
                amount_of_user_asked_queries += amount_of_queries

        print("User were asked to label {} queries".format(
            amount_of_user_asked_queries))

        self.amount_of_user_asked_queries = amount_of_user_asked_queries

    def score(self, X, y):
        return self.amount_of_user_asked_queries


active_learner = Estimator()

grid = RandomizedSearchCV(active_learner,
                          param_distribution_list,
                          n_iter=3,
                          cv=2,
                          verbose=9999999999999999999999999999999999)

evolutionary_search = EvolutionaryAlgorithmSearchCV(
    estimator=active_learner,
    params=param_distribution_list,
    verbose=True,
    cv=2,
    population_size=5,
    gene_mutation_prob=0.10,
    tournament_size=3,
    generations_number=10)

# @todo: remove cross validation
iris = load_iris()

search = grid.fit(iris.data, iris.target)
evolutionary_search.fit(iris.data, iris.target)

print(evolutionary_search.best_params_)
print(evolutionary_search.best_score_)
print(
    pd.DataFrame(evolutionary_search.cv_results_).sort_values(
        "mean_test_score", ascending=False).head())
