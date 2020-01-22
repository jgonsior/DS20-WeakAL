import matplotlib
from itertools import chain
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from experiment_setup_lib import load_and_prepare_X_and_Y


class DataStorage:
    def __init__(self, config):
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
        self.nr_queries_per_iteration = config.nr_queries_per_iteration
        self.config = config

    def load_csv(self, dataset_path):
        X, Y, self.label_encoder = load_and_prepare_X_and_Y(self.config)

        # split data
        X_train, self.X_test, Y_train, self.Y_test = train_test_split(
            X, Y, test_size=self.config.test_fraction)

        # split training data into labeled and unlabeled dataset
        self.X_train_labeled, self.X_train_unlabeled, self.Y_train_labeled, self.Y_train_unlabeled = train_test_split(
            X_train, Y_train, test_size=1 - self.config.start_set_size)

        self._print_data_segmentation()

        self.X_train_unlabeled_cluster_indices = {}

        self.prepare_fake_iteration_zero()

    def prepare_fake_iteration_zero(self):
        # fake iteration zero where we add the given ground truth labels all at once
        original_X_train_labeled = self.X_train_labeled
        original_Y_train_labeled = self.Y_train_labeled

        self.X_train_labeled = np.ndarray(
            shape=(0, original_X_train_labeled.shape[1]))
        self.Y_train_labeled = np.array([], dtype='int64')

        # this one is a bit tricky:
        # we merge both back together here -> but solely for the purpose of using them as the first oracle query down below
        self.X_train_unlabeled = np.concatenate(
            (original_X_train_labeled, self.X_train_unlabeled))
        self.Y_train_unlabeled = np.append(original_Y_train_labeled,
                                           self.Y_train_unlabeled)
        self.ground_truth_indices = [
            i for i in range(0, len(original_Y_train_labeled))
        ]

        self.Y_train_strong_labels = np.array(original_Y_train_labeled)  # copy

    def _print_data_segmentation(self):
        len_train_labeled = len(self.X_train_labeled)
        len_train_unlabeled = len(self.X_train_unlabeled)
        len_test = len(self.X_test)

        len_total = len_train_unlabeled + len_train_labeled + len_test

        print("size of train  labeled set: %i = %1.2f" %
              (len_train_labeled, len_train_labeled / len_total))
        print("size of train unlabeled set: %i = %1.2f" %
              (len_train_unlabeled, len_train_unlabeled / len_total))
        print("size of test set: %i = %1.2f" %
              (len_test, len_test / len_total))

    def move_labeled_queries(self, X_query, Y_query, query_indices):
        # move new queries from unlabeled to labeled dataset
        self.X_train_labeled = np.append(self.X_train_labeled, X_query, 0)
        self.X_train_unlabeled = np.delete(self.X_train_unlabeled,
                                           query_indices, 0)
        self.Y_train_strong_labels = np.append(
            self.Y_train_strong_labels, self.Y_train_unlabeled[query_indices],
            0)
        self.Y_train_labeled = np.append(self.Y_train_labeled, Y_query)
        self.Y_train_unlabeled = np.delete(self.Y_train_unlabeled,
                                           query_indices, 0)

        # remove indices from all clusters
        for cluster in self.X_train_unlabeled_cluster_indices.keys():
            for indice in query_indices:
                if indice in self.X_train_unlabeled_cluster_indices[cluster]:
                    self.X_train_unlabeled_cluster_indices[cluster].remove(
                        indice)
        # print out the amount of stored data in X_train_unlabeled_cluster_indices
        #  print(
        #  len(
        #  list(
        #  chain(*list(
        #  self.X_train_unlabeled_cluster_indices.values())))))
