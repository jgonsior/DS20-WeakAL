import argparse
import contextlib
import io
import os
import random
import sys
import pandas as pd
import numpy as np
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from scipy.stats import uniform, randint
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
    "output_dir": [standard_config.output_dir],
    "random_seed": [standard_config.random_seed],
    "test_fraction": [standard_config.test_fraction],
    "sampling": ['random', 'comittee', 'boundary'],
    "cluster": [
        'dummy', 'random', 'RoundRobin', 'MostUncertain_lc',
        'MostUncertain_max_margin', 'MostUncertain_entropy'
    ],
    "nr_learning_iterations": [1000000000],
    "nr_queries_per_iteration":
    np.random.randint(1, 1000, size=100),
    "start_set_size":
    np.random.uniform(0, 0.5, size=100),
    "minimum_test_accuracy_before_recommendations":
    np.random.uniform(0.5, 1, size=100),
    "uncertainty_recommendation_certainty_threshold":
    np.random.uniform(0.5, 1, size=100),
    "uncertainty_recommendation_ratio": [1 / 10, 1 / 100, 1 / 1000, 1 / 10000],
    "snuba_lite_minimum_heuristic_accuracy":
    np.random.uniform(0.5, 1, size=100),
    "cluster_recommendation_minimum_cluster_unity_size":
    np.random.uniform(0.5, 1, size=100),
    "cluster_recommendation_ratio_labeled_unlabeled":
    np.random.uniform(0.5, 1, size=100),
    "with_uncertainty_recommendation": [True, False],
    "with_cluster_recommendation": [True, False],
    "with_snuba_lite": [False],
    "plot": [True],
}

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
        pass

    def fit(self, X, y, **kwargs):
        pass

    def score(self, X, y):
        return np.random.uniform(0, 1, size=1)[0]


active_learner = Estimator()
grid = RandomizedSearchCV(active_learner,
                          param_grid,
                          n_iter=3,
                          verbose=9999999999999999999999999999999999)

evolutionary_search = EvolutionaryAlgorithmSearchCV(estimator=active_learner,
                                                    params=param_grid,
                                                    verbose=True,
                                                    population_size=5,
                                                    gene_mutation_prob=0.10,
                                                    tournament_size=3,
                                                    generations_number=10)

dataStorage = DataStorage(standard_config.random_seed)
dataStorage.load_csv(standard_config.dataset_path)
search = grid.fit(dataStorage.X, dataStorage.Y)
evolutionary_search.fit(dataStorage.X, dataStorage.Y)

print(evolutionary_search.best_params_)
print(evolutionary_search.best_score_)
print(
    pd.DataFrame(evolutionary_search.cv_results_).sort_values(
        "mean_test_score", ascending=False).head())
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
