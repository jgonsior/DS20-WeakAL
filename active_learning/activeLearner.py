import abc
import collections
import itertools
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from .experiment_setup_lib import (
    conf_matrix_and_acc,
    get_single_al_run_stats_row,
    get_single_al_run_stats_table_header,
    log_it,
)


class ActiveLearner:
    def __init__(
        self,
        RANDOM_SEED,
        dataset_storage,
        cluster_strategy,
        N_JOBS,
        NR_LEARNING_ITERATIONS,
        NR_QUERIES_PER_ITERATION,
        oracle,
        clf,
        weak_supervision_label_sources=[],
    ):
        self.data_storage = dataset_storage
        self.NR_LEARNING_ITERATIONS = NR_LEARNING_ITERATIONS
        self.nr_queries_per_iteration = NR_QUERIES_PER_ITERATION
        self.clf = clf

        self.metrics_per_al_cycle = {
            "test_acc": [],
            "test_conf_matrix": [],
            "train_acc": [],
            "train_conf_matrix": [],
            "query_length": [],
            "recommendation": [],
            "labels_indices": [],
        }

        self.cluster_strategy = cluster_strategy
        self.amount_of_user_asked_queries = 0
        self.oracle = oracle
        self.weak_supervision_label_sources = weak_supervision_label_sources

    @abc.abstractmethod
    def calculate_next_query_indices(self, X_train_unlabeled_cluster_indices, *args):
        pass

    def fit_clf(self):
        self.clf.fit(
            self.data_storage.X_train_labeled,
            self.data_storage.Y_train_labeled[0],
            sample_weight=compute_sample_weight(
                "balanced", self.data_storage.Y_train_labeled[0]
            ),
        )

    def calculate_pre_metrics(self, X_query, Y_query):
        pass

    def calculate_post_metrics(self, X_query, Y_query):

        conf_matrix, acc = conf_matrix_and_acc(
            self.clf,
            self.data_storage.X_test,
            self.data_storage.Y_test[0],
            self.data_storage.label_encoder,
        )

        self.metrics_per_al_cycle["test_conf_matrix"].append(conf_matrix)
        self.metrics_per_al_cycle["test_acc"].append(acc)

        if self.data_storage.Y_train_unlabeled.shape[0] > 0:
            # experiment
            conf_matrix, acc = conf_matrix_and_acc(
                self.clf,
                self.data_storage.X_train_labeled,
                self.data_storage.Y_train_labeled[0],
                self.data_storage.label_encoder,
            )
        else:
            conf_matrix, acc = None, 0

        self.metrics_per_al_cycle["train_conf_matrix"].append(conf_matrix)
        self.metrics_per_al_cycle["train_acc"].append(acc)

    def get_newly_labeled_data(self):
        X_train_unlabeled_cluster_indices = self.cluster_strategy.get_cluster_indices(
            clf=self.clf, nr_queries_per_iteration=self.nr_queries_per_iteration
        )

        # ask strategy for new datapoint
        query_indices = self.calculate_next_query_indices(
            X_train_unlabeled_cluster_indices
        )

        X_query = self.data_storage.X_train_unlabeled.loc[query_indices]

        # ask oracle for new query
        Y_query = self.oracle.get_labeled_samples(query_indices, self.data_storage)
        return X_query, Y_query, query_indices

    def learn(
        self,
        MINIMUM_TEST_ACCURACY_BEFORE_RECOMMENDATIONS,
        ALLOW_RECOMMENDATIONS_AFTER_STOP,
        USER_QUERY_BUDGET_LIMIT,
        **kwargs,
    ):
        log_it(self.data_storage.label_encoder.classes_)
        log_it("Used Hyperparams:")
        log_it(vars(self))
        log_it(locals())

        log_it(get_single_al_run_stats_table_header())

        self.start_set_size = len(self.data_storage.ground_truth_indices)
        early_stop_reached = False

        for i in range(0, self.NR_LEARNING_ITERATIONS):
            # try to actively get at least this amount of data, but if there is only less data available that's just fine
            if (
                self.data_storage.X_train_unlabeled.shape[0]
                < self.nr_queries_per_iteration
            ):
                self.nr_queries_per_iteration = self.data_storage.X_train_unlabeled.shape[
                    0
                ]
            if self.nr_queries_per_iteration == 0:
                break

            # first iteration - add everything from ground truth
            if i == 0:
                query_indices = self.data_storage.ground_truth_indices
                X_query = self.data_storage.X_train_unlabeled.loc[query_indices]
                Y_query = self.data_storage.Y_train_unlabeled.loc[query_indices]

                recommendation_value = "G"
                Y_query_strong = None
            else:
                X_query = None

                if (
                    self.metrics_per_al_cycle["test_acc"][-1]
                    > MINIMUM_TEST_ACCURACY_BEFORE_RECOMMENDATIONS
                ):
                    # iterate over existing WS sources
                    for labelSource in self.weak_supervision_label_sources:
                        (
                            X_query,
                            Y_query,
                            query_indices,
                            recommendation_value,
                        ) = labelSource.get_labeled_samples()

                        if X_query is not None:
                            break

                if early_stop_reached and X_query is None:
                    break

                if X_query is None:
                    # ask oracle for some new labels
                    X_query, Y_query, query_indices = self.get_newly_labeled_data()
                    recommendation_value = "A"
                    self.amount_of_user_asked_queries += len(Y_query)

            Y_query = Y_query.assign(source=recommendation_value)

            self.metrics_per_al_cycle["recommendation"].append(recommendation_value)
            self.metrics_per_al_cycle["query_length"].append(len(Y_query))
            self.metrics_per_al_cycle["labels_indices"].append(str(query_indices))

            self.data_storage.move_labeled_queries(X_query, Y_query, query_indices)

            self.calculate_pre_metrics(X_query, Y_query)

            # retrain CLASSIFIER
            self.fit_clf()

            self.calculate_post_metrics(X_query, Y_query)

            log_it(
                get_single_al_run_stats_row(
                    i,
                    self.data_storage.X_train_labeled.shape[0],
                    self.data_storage.X_train_unlabeled.shape[0],
                    self.metrics_per_al_cycle,
                )
            )

            if self.amount_of_user_asked_queries > USER_QUERY_BUDGET_LIMIT:
                early_stop_reached = True
                log_it("Budget exhausted")
                if not ALLOW_RECOMMENDATIONS_AFTER_STOP:
                    break

        return (
            self.clf,
            self.metrics_per_al_cycle,
            self.data_storage.Y_train_labeled,
        )
