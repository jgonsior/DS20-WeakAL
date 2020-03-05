import random
from itertools import chain

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiment_setup_lib import log_it


class DataStorage:
    def __init__(self, random_seed):
        if random_seed != -1:
            np.random.seed(random_seed)
            random.seed(random_seed)

    def set_training_data(self,
                          X_train,
                          Y_train,
                          label_encoder,
                          test_fraction,
                          start_set_size,
                          X_test=None,
                          Y_test=None):
        # split training data into labeled and unlabeled dataset
        # ensure that at least one sample of each label is present!
        self.X_train_labeled, self.X_train_unlabeled, self.Y_train_labeled, self.Y_train_unlabeled = train_test_split(
            X_train, Y_train, train_size=start_set_size)

        Y_train_labeled_set = set(self.Y_train_labeled[0].to_numpy())

        if len(Y_train_labeled_set) < len(label_encoder.classes_):
            # move more data here from the classes not present
            for class_not_present in range(0, len(label_encoder.classes_)):
                if class_not_present in Y_train_labeled_set:
                    continue
                Y_not_present = self.Y_train_unlabeled[
                    self.Y_train_unlabeled[0] == class_not_present].iloc[0:1]
                #  print(Y_not_present)
                #  print(Y_not_present.index)
                X_not_present = self.X_train_unlabeled.loc[Y_not_present.index]

                #  print(Y_not_present)
                #  print(X_not_present)

                self.X_train_labeled = self.X_train_labeled.append(
                    X_not_present)
                self.X_train_unlabeled = self.X_train_unlabeled.drop(
                    X_not_present.index)

                self.Y_train_labeled = self.Y_train_labeled.append(
                    Y_not_present)
                self.Y_train_unlabeled = self.Y_train_unlabeled.drop(
                    Y_not_present.index)

        self._print_data_segmentation()

        self.X_train_unlabeled_cluster_indices = {}
        self.prepare_fake_iteration_zero()
        log_it(self.X_train_labeled.shape)
        self.label_encoder = label_encoder
        self.X_test = X_test
        self.Y_test = Y_test

    def prepare_fake_iteration_zero(self):
        # fake iteration zero where we add the given ground truth labels all at once
        original_X_train_labeled = self.X_train_labeled
        original_Y_train_labeled = self.Y_train_labeled

        self.X_train_labeled = pd.DataFrame(
            columns=original_X_train_labeled.columns, dtype=float)
        self.Y_train_labeled = pd.DataFrame(
            columns=original_Y_train_labeled.columns, dtype=int)

        # this one is a bit tricky:
        # we merge both back together here -> but solely for the purpose of using them as the first oracle query down below
        self.X_train_unlabeled = pd.concat(
            [original_X_train_labeled, self.X_train_unlabeled])
        self.Y_train_unlabeled = pd.concat(
            [original_Y_train_labeled, self.Y_train_unlabeled])
        self.ground_truth_indices = original_X_train_labeled.index.tolist()

        self.Y_train_strong_labels = pd.DataFrame.copy(
            original_Y_train_labeled)

    def _print_data_segmentation(self):
        len_train_labeled = len(self.X_train_labeled)
        len_train_unlabeled = len(self.X_train_unlabeled)
        #  len_test = len(self.X_test)

        len_total = len_train_unlabeled + len_train_labeled  #+ len_test

        log_it("size of train  labeled set: %i = %1.2f" %
               (len_train_labeled, len_train_labeled / len_total))
        log_it("size of train unlabeled set: %i = %1.2f" %
               (len_train_unlabeled, len_train_unlabeled / len_total))

    def move_labeled_queries(self, X_query, Y_query, query_indices):
        # move new queries from unlabeled to labeled dataset
        self.X_train_labeled = self.X_train_labeled.append(X_query)
        self.X_train_unlabeled = self.X_train_unlabeled.drop(query_indices)

        self.Y_train_strong_labels = self.Y_train_strong_labels.append(
            self.Y_train_unlabeled.loc[query_indices])

        self.Y_train_labeled = self.Y_train_labeled.append(Y_query)
        self.Y_train_unlabeled = self.Y_train_unlabeled.drop(query_indices)

        # remove indices from all clusters in unlabeled and add to labeled
        for cluster_id in self.X_train_unlabeled_cluster_indices.keys():
            list_to_be_removed_and_appended = []
            for indice in query_indices:
                if indice in self.X_train_unlabeled_cluster_indices[
                        cluster_id]:
                    list_to_be_removed_and_appended.append(indice)

            # don't change a list you're iterating over!
            for indice in list_to_be_removed_and_appended:
                self.X_train_unlabeled_cluster_indices[cluster_id].remove(
                    indice)
                self.X_train_labeled_cluster_indices[cluster_id].append(indice)

        # remove possible empty clusters
        self.X_train_unlabeled_cluster_indices = {
            k: v
            for k, v in self.X_train_unlabeled_cluster_indices.items()
            if len(v) != 0
        }
        #  print out the amount of stored data in X_train_unlabeled_cluster_indices
        #  log_it(
        #  len(
        #  list(
        #  chain(*list(
        #  self.X_train_unlabeled_cluster_indices.values())))))
