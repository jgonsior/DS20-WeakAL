import abc
import argparse
import pickle
import random
import sys
from collections import defaultdict
from pprint import pprint

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from experiment_setup_lib import (classification_report_and_confusion_matrix,
                                  print_data_segmentation)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight


class ActiveLearner:
    def __init__(self, config):
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

        self.nr_learning_iterations = config.nr_learning_iterations
        self.nr_queries_per_iteration = config.nr_queries_per_iteration

        self.best_hyper_parameters = {
            'random_state': config.random_seed,
            'n_jobs': config.cores
        }

        # it's a list because of committee (in all other cases it's just one classifier)
        self.clf_list = [RandomForestClassifier(**self.best_hyper_parameters)]

        self.query_accuracy_list = []

        self.metrics_per_al_cycle = {
            'test_data_metrics': [[] for clf in self.clf_list],
            'train_labeled_data_metrics': [[] for clf in self.clf_list],
            'train_unlabeled_data_metrics': [[] for clf in self.clf_list],
            'train_unlabeled_class_distribution':
            [[] for clf in self.clf_list],
            'stop_certainty_list': [],
            'stop_stddev_list': [],
            'stop_accuracy_list': [],
            'query_length': [],
            'recommendation': []
        }

        self.len_queries = self.nr_learning_iterations * self.nr_queries_per_iteration
        self.config = config

    def set_data(self, X_train_labeled, Y_train_labeled, X_train_unlabeled,
                 Y_train_unlabeled, X_test, Y_test, label_encoder):
        self.X_train_labeled = X_train_labeled
        self.Y_train_labeled = Y_train_labeled
        self.X_train_unlabeled = X_train_unlabeled
        self.Y_train_unlabeled = Y_train_unlabeled
        self.X_test = X_test
        self.Y_test = Y_test

        self.label_encoder = label_encoder

        self.classifier_classes = [
            i for i in range(0, len(label_encoder.classes_))
        ]

    def calculate_stopping_criteria_stddev(self):
        accuracy_list = self.query_accuracy_list
        k = 5

        if len(accuracy_list) < k:
            self.metrics_per_al_cycle['stop_stddev_list'].append(float('NaN'))

        k_list = accuracy_list[-k:]
        stddev = np.std(k_list)
        self.metrics_per_al_cycle['stop_stddev_list'].append(stddev)

    def calculate_stopping_criteria_accuracy(self):
        # we use the accuracy ONLY for the current selected query
        self.metrics_per_al_cycle['stop_accuracy_list'].append(
            self.query_accuracy_list[-1])

    def calculate_stopping_criteria_certainty(self):
        Y_train_unlabeled_pred = self.clf_list[0].predict(
            self.X_train_unlabeled)
        Y_train_unlabeled_pred_proba = self.clf_list[0].predict_proba(
            self.X_train_unlabeled)

        # don't ask
        test = pd.Series(Y_train_unlabeled_pred)
        test1 = pd.Series(self.classifier_classes)

        indices = test.map(lambda x: np.where(test1 == x)[0][0]).tolist()

        class_certainties = [
            Y_train_unlabeled_pred_proba[i][indices[i]]
            for i in range(len(Y_train_unlabeled_pred_proba))
        ]

        result = np.min(class_certainties)
        self.metrics_per_al_cycle['stop_certainty_list'].append(result)

    @abc.abstractmethod
    def calculate_next_query_indices(self, *args):
        pass

    def fit_clf(self):
        self.clf_list[0].fit(self.X_train_labeled,
                             self.Y_train_labeled,
                             sample_weight=compute_sample_weight(
                                 'balanced', self.Y_train_labeled))

    def calculate_current_metrics(self, X_query, Y_query):
        # calculate for stopping criteria the accuracy of the prediction for the selected queries
        self.query_accuracy_list.append(
            accuracy_score(Y_query, self.clf_list[0].predict(X_query)))

        for i, clf in enumerate(self.clf_list):
            metrics = classification_report_and_confusion_matrix(
                clf,
                self.X_test,
                self.Y_test,
                self.config,
                self.label_encoder,
                output_dict=True)

            self.metrics_per_al_cycle['test_data_metrics'][i].append(metrics)

            metrics = classification_report_and_confusion_matrix(
                clf,
                self.X_train_labeled,
                self.Y_train_labeled,
                self.config,
                self.label_encoder,
                output_dict=True)

            self.metrics_per_al_cycle['train_labeled_data_metrics'][i].append(
                metrics)

            metrics = classification_report_and_confusion_matrix(
                clf,
                self.X_train_unlabeled,
                self.Y_train_unlabeled,
                self.config,
                self.label_encoder,
                output_dict=True)

            self.metrics_per_al_cycle['train_unlabeled_data_metrics'][
                i].append(metrics)

            train_unlabeled_class_distribution = defaultdict(int)

            for label in self.label_encoder.inverse_transform(Y_query):
                train_unlabeled_class_distribution[label] += 1

            self.metrics_per_al_cycle['train_unlabeled_class_distribution'][
                i].append(train_unlabeled_class_distribution)

    def move_labeled_queries(self, X_query, Y_query, query_indices):
        # move new queries from unlabeled to labeled dataset
        self.X_train_labeled = np.append(self.X_train_labeled, X_query, 0)
        self.X_train_unlabeled = np.delete(self.X_train_unlabeled,
                                           query_indices, 0)

        self.Y_train_labeled = np.append(self.Y_train_labeled, Y_query)
        self.Y_train_unlabeled = np.delete(self.Y_train_unlabeled,
                                           query_indices, 0)

    def increase_labeled_dataset(self):

        # ask strategy for new datapoint
        query_indices = self.calculate_next_query_indices()
        X_query = self.X_train_unlabeled[query_indices]

        # ask oracle for new query
        Y_query = self.Y_train_unlabeled[query_indices]

        return X_query, Y_query, query_indices

    def certain_recommendation(self):
        # calculate certainties for all of X_train_unlabeled
        certainties = self.clf_list[0].predict_proba(self.X_train_unlabeled)

        recommendation_certainty_threshold = 0.9
        recommendation_ratio = 1 / 100
        #  recommendation_ratio = 1 / 100000000000000000000000000

        amount_of_certain_labels = np.count_nonzero(
            np.where(
                np.max(certainties, 1) > recommendation_certainty_threshold))

        if amount_of_certain_labels > len(
                self.X_train_unlabeled) * recommendation_ratio:
            certain_indices = np.where(
                np.max(certainties, 1) > recommendation_certainty_threshold)
            certain_X = self.X_train_unlabeled[certain_indices]
            recommended_labels = self.clf_list[0].predict(certain_X)

            self.metrics_per_al_cycle['recommendation'].append(1)
            return certain_X, recommended_labels, certain_indices
        else:
            self.metrics_per_al_cycle['recommendation'].append(0)
            return None, None, None

    def snuba_lite_recommendation(self):

        X_weak = Y_weak = weak_indices = None
        recommendation_value = 0
        # @todo prevent snuba_lite from relearning based on itself (so only "strong" labels are being used for weak labeling)

        # for each label and each feature (or feature combination) generate small shallow decision tree -> is it a good idea to limit the amount of used features?!
        accuracies = []
        heuristics = []
        # generated heuristics should only being applied to small subset (which one?)
        # balance jaccard and f1_measure (coverage + accuracy)
        for clf_class in self.classifier_classes:
            # create training and test data set out of current available training/test data
            X_temp = self.X_train_labeled

            # do one vs rest
            Y_temp = np.array(self.Y_train_labeled)  # works like a copy
            Y_temp[Y_temp != clf_class] = -1

            X_temp_train, X_temp_test, Y_temp_train, Y_temp_test = train_test_split(
                X_temp, Y_temp)

            heuristic = DecisionTreeClassifier(max_depth=2)
            heuristic.fit(X_temp_train, Y_temp_train)

            heuristics.append(heuristic)

            accuracies.append(
                accuracy_score(Y_temp_test, heuristic.predict(X_temp_test)))

        # if accuracy of decision tree is high enough -> take recommendation
        minimum_heuristic_accuracy = 0.2

        if np.max(accuracies) > minimum_heuristic_accuracy:
            highest_heuristic = heuristics[np.argmax(accuracies)]

            probabilities = highest_heuristic.predict_proba(
                self.X_train_unlabeled)

            # filter out labels where one-vs-rest heuristic is sure that sample is of label L
            weak_indices = np.where(
                np.argmax(probabilities, 1) > minimum_heuristic_accuracy)

            if weak_indices[0].size > 0:
                X_weak = self.X_train_unlabeled[weak_indices]
                Y_weak = self.Y_train_unlabeled[weak_indices]
                recommendation_value = 2
            else:
                weak_indices = None

        self.metrics_per_al_cycle['recommendation'].append(
            recommendation_value)
        return X_weak, Y_weak, weak_indices

    def learn(self):
        print_data_segmentation(self.X_train_labeled, self.X_train_unlabeled,
                                self.X_test, self.len_queries)

        print(
            "Iteration: {:>3} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>3}"
            .format("I", "L", "U", "Q", "Te", "L", "U", "SC", "SS", "SA",
                    "CR"))

        print("Iteration:  -1 {0:6,d} {1:6,d}".format(
            self.X_train_labeled.shape[0], self.X_train_unlabeled.shape[0]))

        for i in range(0, self.nr_learning_iterations):
            if self.X_train_unlabeled.shape[0] < self.nr_queries_per_iteration:
                print(self.X_train_unlabeled.shape)
                break

            # retrain classifier
            self.fit_clf()

            X_query, Y_query, query_indices = self.certain_recommendation()

            if X_query is None:
                X_query, Y_query, query_indices = self.snuba_lite_recommendation(
                )

            if X_query is None:
                # ask oracle for some "hard data"
                X_query, Y_query, query_indices = self.increase_labeled_dataset(
                )

            if len(X_query) == 0:
                print("hui")
                print(self.metrics_per_al_cycle['recommendation'][-1])

            self.move_labeled_queries(X_query, Y_query, query_indices)
            self.metrics_per_al_cycle['query_length'].append(len(Y_query))

            #  print("Y_train_labeled", self.Y_train_labeled.shape)
            #  print("Y_train_unlabeled", self.Y_train_unlabeled.shape)
            #  print("Y_test", self.Y_test.shape)
            #  print("Y_query", Y_query.shape)

            #  print("X_train_labeled", self.X_train_labeled.shape)
            #  print("X_train_unlabeled", self.X_train_unlabeled.shape)
            #  print("X_test", self.X_test.shape)
            #  print("X_query", X_query.shape)

            # calculate new metrics
            self.calculate_current_metrics(X_query, Y_query)

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

            print(
                "Iteration: {:3,d} {:6,d} {:6,d} {:6,d} {:6.1%} {:6.1%} {:6.1%} {:6.1%} {:6.1%} {:6.1%} {:>3}"
                .format(
                    i,
                    self.X_train_labeled.shape[0],
                    self.X_train_unlabeled.shape[0],
                    self.metrics_per_al_cycle['query_length'][-1],
                    self.metrics_per_al_cycle['test_data_metrics'][0][-1][0]
                    ['accuracy'],
                    self.metrics_per_al_cycle['train_labeled_data_metrics'][0]
                    [-1][0]['accuracy'],
                    self.metrics_per_al_cycle['train_unlabeled_data_metrics']
                    [0][-1][0]['accuracy'],
                    self.metrics_per_al_cycle['stop_certainty_list'][-1],
                    self.metrics_per_al_cycle['stop_stddev_list'][-1],
                    self.metrics_per_al_cycle['stop_accuracy_list'][-1],
                    self.metrics_per_al_cycle['recommendation'][-1],
                ))

        # in case we specified more queries than we have data
        self.nr_learning_iterations = i
        self.len_queries = self.nr_learning_iterations * self.nr_queries_per_iteration

        #  print(self.metrics_per_al_cycle)

        return self.clf_list, self.metrics_per_al_cycle
