import abc
import argparse
import pickle
import random
import sys
from collections import defaultdict
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

import functions


class ActiveLearner:
    def __init__(self, config):
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

        # list of possible labels for classification

        self.nr_learning_iterations = config.nLearningIterations
        self.nr_queries_per_iteration = config.nQueriesPerIteration

        self.best_hyper_parameters = {
            **functions.get_best_hyper_params('RF'), 'random_state':
            config.random_seed,
            'n_jobs': config.cores
        }

        # it's a list because of committee (in all other cases it's just one classifier)
        self.clf_list = [RandomForestClassifier(**self.best_hyper_parameters)]

        self.accuracy_list = []
        self.query_accuracy_list = []
        #  self.class_accuracy_list = [[] for clf in self.clf_list]

        self.metrics_per_time = {
            'test_data': [[] for clf in self.clf_list],
            'train_data': [[] for clf in self.clf_list],
            'query_data': [[] for clf in self.clf_list],
            'query_set_distribution': [[] for clf in self.clf_list],
            'stop_certainty_list': [],
            'stop_stddev_list': [],
            'stop_accuracy_list': [],
            'queried_sheet': [],
            'queried_length': [],
        }

        self.len_queries = self.nr_learning_iterations * self.nr_queries_per_iteration

        # reading data of files build by data builder
        (
            (self.X_train, self.Y_train),
            (self.X_query, self.Y_query),
            (self.X_test, self.Y_test),
            (self.X_query_spreadsheets),
        ), labelBinarizer, = functions.load_query_data(
            featuresPath=config.features,
            metaPath=config.meta,
            start_size=config.start_size,
            merged_labels=config.mergedLabels,
            random_state=config.random_seed)

        # store dataframes for later on results
        self.X_train_orig = self.X_train.copy()
        self.X_query_orig = self.X_query.copy()
        self.X_test_orig = self.X_test.copy()

        self.Y_train_orig = np.copy(self.Y_train)
        self.Y_query_orig = np.copy(self.Y_query)
        self.Y_test_orig = np.copy(self.Y_test)

        self.labelBinarizer = labelBinarizer
        self.classifier_classes = [
            i for i in range(0, len(labelBinarizer.classes_))
        ]
        self.config = config

    def check_stopping_criteria_stddev(self, accuracy_list, k):
        if (len(accuracy_list) < k): return False

        k_list = accuracy_list[-k:]
        stddev = np.std(k_list)
        self.metrics_per_time['stop_stddev_list'].append(stddev)

    def check_stopping_criteria_accuracy(self, query_accuracy):
        self.metrics_per_time['stop_accuracy_list'].append(query_accuracy)

    def check_stopping_criteria_certainty(self, clf, X_temp, Y_temp):
        Y_temp_pred = clf.predict(X_temp)
        Y_temp_proba = clf.predict_proba(X_temp)

        test = pd.Series(Y_temp_pred)
        test1 = pd.Series(self.classifier_classes)

        indices = test.map(lambda x: np.where(test1 == x)[0][0]).tolist()

        class_certainties = [
            Y_temp_proba[i][indices[i]] for i in range(len(Y_temp_proba))
        ]

        result = np.min(class_certainties)
        self.metrics_per_time['stop_certainty_list'].append(result)

    @abc.abstractmethod
    def retrieve_query_indices(self, *args):
        pass

    def fitClf(self):
        self.clf_list[0].fit(self.X_train,
                             self.Y_train,
                             sample_weight=compute_sample_weight(
                                 'balanced', self.Y_train))

    def calculateMetrics(self):

        current_clf_list = self.clf_list
        for i, clf in enumerate(current_clf_list):
            metrics = functions.clf_results(
                clf,
                self.X_test,
                self.Y_test,
                output_dict=True,
                n_jobs=self.config.cores,
                target_names=self.labelBinarizer.classes_)

            self.metrics_per_time['test_data'][i].append(metrics)
            self.accuracy_list.append(metrics[0]['accuracy'])

        for i, clf in enumerate(current_clf_list):
            metrics = functions.clf_results(
                clf,
                self.X_train,
                self.Y_train,
                output_dict=True,
                n_jobs=self.config.cores,
                target_names=self.labelBinarizer.classes_)
            self.metrics_per_time['train_data'][i].append(metrics)

        for i, clf in enumerate(current_clf_list):
            metrics = functions.clf_results(
                clf,
                self.X_query,
                self.Y_query,
                output_dict=True,
                n_jobs=self.config.cores,
                target_names=self.labelBinarizer.classes_)
            self.metrics_per_time['query_data'][i].append(metrics)

    def get_Ys_for_indice(self, query_indices):
        return self.Y_query[query_indices]

    def move_new_labels_to_training_data(self, query_indices):
        # move new labels to training data for next classification
        rows = self.get_Ys_for_indice(query_indices)
        self.Y_train = np.append(self.Y_train, rows)
        self.Y_query = np.delete(self.Y_query, query_indices, 0)

    def move_new_queries_to_training_data(self, query_indices):
        # move new queries to training data for next classification
        rows = self.X_query.iloc()[query_indices]
        self.X_train = self.X_train.append(rows, ignore_index=True)
        self.X_query.drop(rows.index, inplace=True)

    def run(self):
        functions.print_data_segmentation(self.X_train, self.X_query,
                                          self.X_test, self.len_queries)

        for i in range(0, self.nr_learning_iterations):
            if (self.X_query.shape[0] < self.nr_queries_per_iteration) or len(
                    self.X_query_spreadsheets) == 0:
                break

            print("Iteration: %d" % i)
            pprint(self.X_query.shape[0])

            self.fitClf()
            self.calculateMetrics()

            # retrieve queries of classifier
            query_indices = self.retrieve_query_indices()

            query_set_accuracy = functions.calculate_current_accuracy(
                self.clf_list[0],
                self.X_query.iloc()[query_indices],
                self.get_Ys_for_indice(query_indices))

            self.query_accuracy_list.append(query_set_accuracy)
            if isinstance(query_indices, tuple):
                length = len(query_indices[0])
            else:
                length = len(query_indices)

            self.metrics_per_time['queried_length'].append(length)

            for j, clf in enumerate(self.clf_list):
                query_set_distribution = defaultdict(int)

                for label in self.labelBinarizer.inverse_transform(
                        self.get_Ys_for_indice(query_indices)):
                    query_set_distribution[label] += 1

                self.metrics_per_time['query_set_distribution'][j].append(
                    query_set_distribution)

            if i >= 5:
                self.check_stopping_criteria_accuracy(
                    self.query_accuracy_list[-1]
                )  # sollte nur die accuracy für das gerade eben gelabelte query pair erhalten, nicht für die noch verbleibenden queries
                self.check_stopping_criteria_stddev(self.query_accuracy_list,
                                                    5)
                self.check_stopping_criteria_certainty(clf, self.X_query,
                                                       self.Y_query)

            self.move_new_labels_to_training_data(query_indices)
            self.move_new_queries_to_training_data(query_indices)

        # in case we specified more queries than we have data
        self.nr_learning_iterations = i
        self.len_queries = self.nr_learning_iterations * self.nr_queries_per_iteration

        pprint(self.metrics_per_time)
        with open(
                self.config.output + '/' + self.config.strategy + '_' +
                str(self.config.start_size) + '_' +
                str(self.config.nQueriesPerIteration) + '.pickle', 'wb') as f:
            pickle.dump(self.metrics_per_time, f, pickle.HIGHEST_PROTOCOL)

        clf_active = self.clf_list[0]

        clf_passive_full = RandomForestClassifier(**self.best_hyper_parameters)
        clf_passive_starter = RandomForestClassifier(
            **self.best_hyper_parameters)

        functions.print_clf_comparison(
            clf_active=clf_active,
            clf_passive_starter=clf_passive_starter,
            clf_passive_full=clf_passive_full,
            len_active=self.len_queries,
            features_path=self.config.features,
            meta_path=self.config.meta,
            classifier_classes=self.classifier_classes,
            target_names=self.labelBinarizer.classes_,
            start_size=self.config.start_size,
            merged_labels=self.config.mergedLabels,
            random_state=self.config.random_seed,
            X_train_orig=self.X_train_orig,
            X_query_orig=self.X_query_orig,
            X_test_orig=self.X_test_orig,
            Y_train_orig=self.Y_train_orig,
            Y_query_orig=self.Y_query_orig,
            Y_test_orig=self.Y_test_orig)
