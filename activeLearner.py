import abc
import argparse
import collections
import itertools
import logging
import pickle
import random
import sys
from collections import defaultdict
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from experiment_setup_lib import (classification_report_and_confusion_matrix,
                                  get_single_al_run_stats_row,
                                  get_single_al_run_stats_table_header)


class ActiveLearner:
    def __init__(self,
                 random_seed,
                 dataset_storage,
                 cluster_strategy,
                 cores,
                 nr_learning_iterations,
                 nr_queries_per_iteration,
                 with_test=True):
        if random_seed != -1:
            np.random.seed(random_seed)
            random.seed(random_seed)

            self.best_hyper_parameters = {
                'random_state': random_seed,
                'n_jobs': cores
            }
        else:
            self.best_hyper_parameters = {'n_jobs': cores}

        self.nr_learning_iterations = nr_learning_iterations
        self.nr_queries_per_iteration = nr_queries_per_iteration

        # it's a list because of committee (in all other cases it's just one classifier)
        self.clf_list = [RandomForestClassifier(**self.best_hyper_parameters)]

        self.query_weak_accuracy_list = []

        self.metrics_per_al_cycle = {
            'test_data_metrics': [[] for clf in self.clf_list],
            'train_labeled_data_metrics': [[] for clf in self.clf_list],
            'train_unlabeled_data_metrics': [[] for clf in self.clf_list],
            'train_unlabeled_class_distribution':
            [[] for clf in self.clf_list],
            'stop_certainty_list': [],
            'stop_stddev_list': [],
            'stop_query_weak_accuracy_list': [],
            'query_strong_accuracy_list': [],
            'query_length': [],
            'recommendation': []
        }

        self.with_test = with_test
        self.cluster_strategy = cluster_strategy
        self.data_storage = dataset_storage

    def calculate_stopping_criteria_stddev(self):
        accuracy_list = self.query_weak_accuracy_list
        k = 5

        if len(accuracy_list) < k:
            self.metrics_per_al_cycle['stop_stddev_list'].append(float('NaN'))

        k_list = accuracy_list[-k:]
        stddev = np.std(k_list)
        self.metrics_per_al_cycle['stop_stddev_list'].append(stddev)

    def calculate_stopping_criteria_accuracy(self):
        # we use the accuracy ONLY for the current selected query
        self.metrics_per_al_cycle['stop_query_weak_accuracy_list'].append(
            self.query_weak_accuracy_list[-1])

    def calculate_stopping_criteria_certainty(self):
        Y_train_unlabeled_pred_proba = self.clf_list[0].predict_proba(
            self.data_storage.X_train_unlabeled.to_numpy())

        result = np.apply_along_axis(entropy, 1, Y_train_unlabeled_pred_proba)
        self.metrics_per_al_cycle['stop_certainty_list'].append(np.max(result))

    @abc.abstractmethod
    def calculate_next_query_indices(self, X_train_unlabeled_cluster_indices,
                                     *args):
        pass

    def fit_clf(self):
        self.clf_list[0].fit(
            self.data_storage.X_train_labeled.to_numpy(),
            self.data_storage.Y_train_labeled[0].to_numpy(),
            sample_weight=compute_sample_weight(
                'balanced', self.data_storage.Y_train_labeled[0].to_numpy()))

    def calculate_pre_metrics(self, X_query, Y_query, Y_query_strong=None):
        # calculate for stopping criteria the accuracy of the prediction for the selected queries

        try:
            self.query_weak_accuracy_list.append(
                accuracy_score(Y_query, self.clf_list[0].predict(X_query)))
        except NotFittedError:
            self.query_weak_accuracy_list.append(1)

    def calculate_post_metrics(self, X_query, Y_query, Y_query_strong=None):
        if Y_query_strong is not None:
            self.metrics_per_al_cycle['query_strong_accuracy_list'].append(
                accuracy_score(Y_query_strong, Y_query))
        else:
            self.metrics_per_al_cycle['query_strong_accuracy_list'].append(
                float('NaN'))

        for i, clf in enumerate(self.clf_list):
            metrics = classification_report_and_confusion_matrix(
                clf,
                self.data_storage.X_train_labeled,
                self.data_storage.Y_train_labeled,
                self.data_storage.label_encoder,
                output_dict=True)

            self.metrics_per_al_cycle['train_labeled_data_metrics'][i].append(
                metrics)

            if self.with_test:
                metrics = classification_report_and_confusion_matrix(
                    clf,
                    self.data_storage.X_test,
                    self.data_storage.Y_test,
                    self.data_storage.label_encoder,
                    output_dict=True)

            # in case we don't have the metrics just store the train results
            self.metrics_per_al_cycle['test_data_metrics'][i].append(metrics)

            if self.data_storage.X_train_unlabeled.shape[0] != 0:
                metrics = classification_report_and_confusion_matrix(
                    clf,
                    self.data_storage.X_train_unlabeled,
                    self.data_storage.Y_train_unlabeled,
                    self.data_storage.label_encoder,
                    output_dict=True)
            else:
                metrics = ({'accuracy': 0}, None)

            self.metrics_per_al_cycle['train_unlabeled_data_metrics'][
                i].append(metrics)

            train_unlabeled_class_distribution = defaultdict(int)

            if isinstance(Y_query, pd.DataFrame):
                for label in self.data_storage.label_encoder.inverse_transform(
                        Y_query[0].to_numpy()):
                    train_unlabeled_class_distribution[label] += 1
            else:
                for label in self.data_storage.label_encoder.inverse_transform(
                        Y_query):
                    train_unlabeled_class_distribution[label] += 1

            self.metrics_per_al_cycle['train_unlabeled_class_distribution'][
                i].append(train_unlabeled_class_distribution)

    def cluster_recommendation(self, minimum_cluster_unity_size,
                               minimum_ratio_labeled_unlabeled):
        certain_X = recommended_labels = certain_indices = None
        cluster_found = False

        # check if the most prominent label for one cluster can be propagated over to the rest of it's cluster
        for cluster_id, cluster_indices in self.data_storage.X_train_labeled_cluster_indices.items(
        ):
            if cluster_id not in self.data_storage.X_train_unlabeled_cluster_indices.keys(
            ):
                continue
            if len(cluster_indices) / len(
                    self.data_storage.X_train_unlabeled_cluster_indices[
                        cluster_id]) > minimum_cluster_unity_size:
                frequencies = collections.Counter(
                    self.data_storage.Y_train_labeled.loc[cluster_indices]
                    [0].tolist())
                if frequencies.most_common(1)[0][1] > len(
                        cluster_indices) * minimum_ratio_labeled_unlabeled:
                    certain_indices = self.data_storage.X_train_unlabeled_cluster_indices[
                        cluster_id]

                    certain_X = self.data_storage.X_train_unlabeled.loc[
                        certain_indices]
                    recommended_labels = [
                        frequencies.most_common(1)[0][0]
                        for _ in certain_indices
                    ]
                    recommended_labels = pd.DataFrame(recommended_labels,
                                                      index=certain_X.index)
                    #  logging.info("Cluster ", cluster_id, certain_indices)
                    cluster_found = True
                    break

        # delete this cluster from the list of possible cluster for the next round
        if cluster_found:
            self.data_storage.X_train_labeled_cluster_indices.pop(cluster_id)
        return certain_X, recommended_labels, certain_indices

    def increase_labeled_dataset(self):
        # dict of cluster -> [X_train_unlabeled_indices]
        X_train_unlabeled_cluster_indices = self.cluster_strategy.get_cluster_indices(
            clf=self.clf_list[0],
            nr_queries_per_iteration=self.nr_queries_per_iteration)

        #  for cluster_id, cluster_indices in X_train_unlabeled_cluster_indices.items(
        #  ):
        #  logging.info(
        #  "Selected cluster ", cluster_id, ":\t",
        #  self.cluster_strategy._entropy(
        #  self.data_storage.Y_train_unlabeled.loc[cluster_indices]))

        # ask strategy for new datapoint
        query_indices = self.calculate_next_query_indices(
            X_train_unlabeled_cluster_indices)

        X_query = self.data_storage.X_train_unlabeled.loc[query_indices]

        # ask oracle for new query
        Y_query = self.data_storage.Y_train_unlabeled.loc[query_indices]
        return X_query, Y_query, query_indices

    def uncertainty_recommendation(self, recommendation_certainty_threshold,
                                   recommendation_ratio):
        # calculate certainties for all of X_train_unlabeled
        certainties = self.clf_list[0].predict_proba(
            self.data_storage.X_train_unlabeled.to_numpy())

        amount_of_certain_labels = np.count_nonzero(
            np.where(
                np.max(certainties, 1) > recommendation_certainty_threshold))

        if amount_of_certain_labels > len(
                self.data_storage.X_train_unlabeled) * recommendation_ratio:

            # for safety reasons I refrain from explaining the following
            certain_indices = [
                j for i, j in enumerate(
                    self.data_storage.X_train_unlabeled.index.tolist()) if
                np.max(certainties, 1)[i] > recommendation_certainty_threshold
            ]

            certain_X = self.data_storage.X_train_unlabeled.loc[
                certain_indices]

            recommended_labels = self.clf_list[0].predict(certain_X.to_numpy())
            # add indices to recommended_labels, could be maybe useful later on?
            recommended_labels = pd.DataFrame(recommended_labels,
                                              index=certain_X.index)

            return certain_X, recommended_labels, certain_indices
        else:
            return None, None, None

    def snuba_lite_recommendation(self, minimum_heuristic_accuracy):

        X_weak = Y_weak = weak_indices = None
        # @todo prevent snuba_lite from relearning based on itself (so only "strong" labels are being used for weak labeling)
        # for each label and each feature (or feature combination) generate small shallow decision tree -> is it a good idea to limit the amount of used features?!
        highest_accuracy = 0
        best_heuristic = None
        best_combination = None
        best_class = None

        combinations = []
        for combination in itertools.combinations(
                list(range(self.data_storage.X_train_labeled.shape[1])), 1):
            combinations.append(combination)

        # generated heuristics should only being applied to small subset (which one?)
        # balance jaccard and f1_measure (coverage + accuracy)
        for clf_class in self.data_storage.label_encoder.classes_:
            for combination in combinations:
                # create training and test data set out of current available training/test data
                X_temp = self.data_storage.X_train_labeled.loc[:, combination]

                # do one vs rest
                Y_temp = self.data_storage.Y_train_labeled.copy()
                Y_temp = Y_temp.replace(
                    self.data_storage.label_encoder.transform([
                        c for c in self.data_storage.label_encoder.classes_
                        if c != clf_class
                    ]), -1)

                X_temp_train, X_temp_test, Y_temp_train, Y_temp_test = train_test_split(
                    X_temp, Y_temp, train_size=0.6)

                heuristic = DecisionTreeClassifier(max_depth=2)
                heuristic.fit(X_temp_train, Y_temp_train)

                accuracy = accuracy_score(Y_temp_test,
                                          heuristic.predict(X_temp_test))
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    best_heuristic = heuristic
                    best_combination = combination
                    best_class = clf_class

        # if accuracy of decision tree is high enough -> take recommendation
        if highest_accuracy > minimum_heuristic_accuracy:
            probabilities = best_heuristic.predict_proba(
                self.data_storage.X_train_unlabeled.loc[:, best_combination].
                to_numpy())

            # filter out labels where one-vs-rest heuristic is sure that sample is of label L
            weak_indices = [
                index for index, proba in zip(
                    self.data_storage.X_train_unlabeled.index, probabilities)
                if np.max(proba) > minimum_heuristic_accuracy
            ]

            if len(weak_indices) > 0:
                logging.info("Snuba mit Klasse " + best_class)
                #  logging.info(weak_indices)
                X_weak = self.data_storage.X_train_unlabeled.loc[weak_indices]
                best_class_encoded = self.data_storage.label_encoder.transform(
                    [best_class])[0]
                Y_weak = [best_class_encoded for _ in weak_indices]
            else:
                weak_indices = None

        return X_weak, Y_weak, weak_indices

    def learn(
        self,
        minimum_test_accuracy_before_recommendations,
        with_cluster_recommendation,
        with_uncertainty_recommendation,
        with_snuba_lite,
        stopping_criteria_uncertainty,
        stopping_criteria_acc,
        stopping_criteria_std,
        allow_recommendations_after_stop,
        cluster_recommendation_minimum_cluster_unity_size=None,
        cluster_recommendation_minimum_ratio_labeled_unlabeled=None,
        uncertainty_recommendation_certainty_threshold=None,
        uncertainty_recommendation_ratio=None,
        snuba_lite_minimum_heuristic_accuracy=None,
    ):
        logging.info(self.data_storage.label_encoder.classes_)
        logging.info("Used Hyperparams:")
        logging.info(vars(self))
        logging.info(locals())

        logging.info(get_single_al_run_stats_table_header())

        self.start_set_size = len(self.data_storage.ground_truth_indices)
        early_stop_reached = False

        for i in range(0, self.nr_learning_iterations):
            # try to actively get at least this amount of data, but if there is only less data available that's just fine
            if self.data_storage.X_train_unlabeled.shape[
                    0] < self.nr_queries_per_iteration:
                self.nr_queries_per_iteration = self.data_storage.X_train_unlabeled.shape[
                    0]

            if self.nr_queries_per_iteration == 0:
                break

            # first iteration - add everything from ground truth
            if i == 0:
                query_indices = self.data_storage.ground_truth_indices
                X_query = self.data_storage.X_train_unlabeled.loc[
                    query_indices]
                Y_query = self.data_storage.Y_train_unlabeled.loc[
                    query_indices]

                recommendation_value = "G"
                Y_query_strong = None
            else:
                X_query = None

                if self.metrics_per_al_cycle['stop_query_weak_accuracy_list'][
                        -1] > minimum_test_accuracy_before_recommendations:
                    if X_query is None and with_cluster_recommendation:
                        X_query, Y_query, query_indices = self.cluster_recommendation(
                            cluster_recommendation_minimum_cluster_unity_size,
                            cluster_recommendation_minimum_ratio_labeled_unlabeled
                        )
                        recommendation_value = "C"

                    if X_query is None and with_uncertainty_recommendation:
                        X_query, Y_query, query_indices = self.uncertainty_recommendation(
                            uncertainty_recommendation_certainty_threshold,
                            uncertainty_recommendation_ratio)
                        recommendation_value = "U"

                    if X_query is None and with_snuba_lite:
                        X_query, Y_query, query_indices = self.snuba_lite_recommendation(
                            snuba_lite_minimum_heuristic_accuracy)
                        recommendation_value = "S"

                if X_query is not None:
                    Y_query_strong = self.data_storage.Y_train_unlabeled.loc[
                        query_indices]
                    #  logging.info(Y_query_strong)
                    #  logging.info(Y_query)

                if early_stop_reached and X_query is None:
                    break
                if X_query is None:
                    # ask oracle for some "hard data"
                    X_query, Y_query, query_indices = self.increase_labeled_dataset(
                    )
                    recommendation_value = "A"
                    Y_query_strong = None

            self.metrics_per_al_cycle['recommendation'].append(
                recommendation_value)
            self.metrics_per_al_cycle['query_length'].append(len(Y_query))

            self.data_storage.move_labeled_queries(X_query, Y_query,
                                                   query_indices)

            #  logging.info("Y_train_labeled", self.data_storage.Y_train_labeled.shape)
            #  logging.info("Y_train_unlabeled", self.data_storage.Y_train_unlabeled.shape)
            #  logging.info("Y_test", self.data_storage.Y_test.shape)
            #  logging.info("Y_query", Y_query.shape)
            #  logging.info("Y_train_strong_labels", self.data_storage.Y_train_strong_labels)
            #  logging.info("indices", query_indices)

            #  logging.info("X_train_labeled", self.data_storage.X_train_labeled.shape)
            #  logging.info("X_train_unlabeled", self.data_storage.X_train_unlabeled.shape)
            #  logging.info("X_test", self.data_storage.X_test.shape)
            #  logging.info("X_query", X_query.shape)

            self.calculate_pre_metrics(X_query,
                                       Y_query,
                                       Y_query_strong=Y_query_strong)

            # retrain classifier
            self.fit_clf()

            # calculate new metrics
            self.calculate_post_metrics(X_query,
                                        Y_query,
                                        Y_query_strong=Y_query_strong)

            if len(self.data_storage.Y_train_unlabeled) != 0:
                self.calculate_stopping_criteria_accuracy()
                self.calculate_stopping_criteria_stddev()
                self.calculate_stopping_criteria_certainty()

            if 'accuracy' not in self.metrics_per_al_cycle[
                    'train_unlabeled_data_metrics'][0][-1][0].keys():
                self.metrics_per_al_cycle['train_unlabeled_data_metrics'][0][
                    -1][0]['accuracy'] = np.sum(
                        self.metrics_per_al_cycle['train_labeled_data_metrics']
                        [0][-1][1].diagonal()) / np.sum(
                            self.metrics_per_al_cycle[
                                'train_labeled_data_metrics'][0][-1][1])

            logging.info(
                get_single_al_run_stats_row(
                    i, self.data_storage.X_train_labeled.shape[0],
                    self.data_storage.X_train_unlabeled.shape[0],
                    self.metrics_per_al_cycle))

            # checking stop criterias
            if self.metrics_per_al_cycle['stop_query_weak_accuracy_list'][
                    -1] > stopping_criteria_acc and self.metrics_per_al_cycle[
                        'stop_stddev_list'][
                            -1] < stopping_criteria_std and self.metrics_per_al_cycle[
                                'stop_certainty_list'][
                                    -1] < stopping_criteria_std:
                early_stop_reached = True
                logging.info("Early stop")
                if not allow_recommendations_after_stop:
                    break

        self.calculate_amount_of_user_asked_queries()

        return self.clf_list, self.metrics_per_al_cycle

    def calculate_amount_of_user_asked_queries(self):
        amount_of_user_asked_queries = self.start_set_size

        for i, amount_of_queries in enumerate(
                self.metrics_per_al_cycle['query_length']):
            if self.metrics_per_al_cycle['recommendation'][i] == "A":
                amount_of_user_asked_queries += amount_of_queries
        self.amount_of_user_asked_queries = amount_of_user_asked_queries
