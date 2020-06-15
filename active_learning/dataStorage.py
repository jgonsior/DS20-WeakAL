import random
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .experiment_setup_lib import log_it


class DataStorage:
    def __init__(self, random_seed):
        if random_seed != -1:
            np.random.seed(random_seed)
            random.seed(random_seed)

    def set_training_data(
        self,
        X_labeled,
        Y_labeled,
        X_unlabeled=None,
        START_SET_SIZE=None,
        TEST_FRACTION=None,
        label_encoder=None,
        hyper_parameters=None,
        X_test=None,
        Y_test=None,
    ):
        """ 
        1. get start_set from X_labeled
        2. if X_unlabeled is None : experiment!
            2.1 if X_test: rest von X_labeled wird X_train_unlabeled
            2.2 if X_test is none: split rest von X_labeled in X_train_unlabeled und X_test
           else (kein experiment):
           X_unlabeled wird X_unlabeled, rest von X_labeled wird X_train_unlabeled_
        
        """

        # separate X_labeled into start_set and labeled _rest

        if START_SET_SIZE == len(X_labeled):
            X_labeled_rest = None
            self.X_train_labeled = X_labeled
            Y_labeled_rest = None
            self.Y_train_labeled = Y_labeled

        else:
            (
                X_labeled_rest,
                self.X_train_labeled,
                Y_labeled_rest,
                self.Y_train_labeled,
            ) = train_test_split(X_labeled, Y_labeled, test_size=START_SET_SIZE)

        # check if the minimum amount of labeled data is present in the start set size
        labels_not_in_start_set = set(range(0, len(label_encoder.classes_)))
        all_label_in_start_set = False

        for Y in self.Y_train_labeled.to_numpy()[0]:
            if Y in labels_not_in_start_set:
                labels_not_in_start_set.remove(Y)
            if len(labels_not_in_start_set) == 0:
                all_label_in_start_set = True
                break

        if not all_label_in_start_set:
            if X_labeled_rest is None:
                print("Please specify at least one labeled example of each class")
                exit(-1)

            # move more data here from the classes not present
            for label in labels_not_in_start_set:
                Y_not_present = Y_labeled_rest[Y_labeled_rest[0] == label].iloc[0:1]

                # iloc not loc because we use the index from numpy
                X_not_present = X_labeled_rest.loc[Y_not_present.index]

                self.X_train_labeled = self.X_train_labeled.append(X_not_present)
                X_labeled_rest = X_labeled_rest.drop(X_not_present.index)

                self.Y_train_labeled = self.Y_train_labeled.append(Y_not_present)
                Y_labeled_rest = Y_labeled_rest.drop(Y_not_present.index)

        if X_unlabeled is not None:
            self.X_train_unlabeled = X_unlabeled
            self.Y_train_unlabeled = pd.DataFrame(
                columns=Y_labeled_rest.columns, dtype=int
            )

            self.X_test = X_labeled_rest
            self.Y_test = Y_labeled_rest
        else:
            # experiment setting!
            # create some fake unlabeled data

            if X_test is not None:
                self.X_train_unlabeled = X_labeled_rest
                self.Y_train_unlabeled = Y_labeled_rest
                self.X_test = X_test
                self.Y_test = Y_test
            else:
                # further split labeled rest for train_test
                (
                    self.X_train_unlabeled,
                    self.X_test,
                    self.Y_train_unlabeled,
                    self.Y_test,
                ) = train_test_split(X_labeled_rest, Y_labeled_rest, TEST_FRACTION)

        Y_train_labeled_set = set(self.Y_train_labeled[0].to_numpy())

        self._print_data_segmentation()

        self.X_train_unlabeled_cluster_indices = {}

        # remove the labeled data from X_train_labeled and merge it with the unlabeled data
        # while preserving the labels
        # and storing the indics of the labeled data
        # so that the first iteration can be a "fake iteration zero" of the AL cycle
        # (metrics will than automatically be calculated for this one too)
        self.prepare_fake_iteration_zero()
        log_it(self.X_train_labeled.shape)
        self.label_encoder = label_encoder

    def prepare_fake_iteration_zero(self):
        # fake iteration zero where we add the given ground truth labels all at once
        original_X_train_labeled = self.X_train_labeled
        original_Y_train_labeled = self.Y_train_labeled

        self.X_train_labeled = pd.DataFrame(
            columns=original_X_train_labeled.columns, dtype=float
        )

        if self.X_train_labeled is not None:
            self.Y_train_labeled = pd.DataFrame(
                columns=original_Y_train_labeled.columns, dtype=int
            )

        # this one is a bit tricky:
        # we merge both back together here -> but solely for the purpose of using them as the first oracle query down below
        self.X_train_unlabeled = pd.concat(
            [original_X_train_labeled, self.X_train_unlabeled]
        )

        self.Y_train_unlabeled = pd.concat(
            [original_Y_train_labeled, self.Y_train_unlabeled]
        )
        self.ground_truth_indices = original_X_train_labeled.index.tolist()

        self.Y_train_strong_labels = pd.DataFrame.copy(original_Y_train_labeled)

    def _print_data_segmentation(self):
        len_train_labeled = len(self.X_train_labeled)
        len_train_unlabeled = len(self.X_train_unlabeled)
        #  len_test = len(self.X_test)

        len_total = len_train_unlabeled + len_train_labeled  # + len_test

        log_it(
            "size of train  labeled set: %i = %1.2f"
            % (len_train_labeled, len_train_labeled / len_total)
        )
        log_it(
            "size of train unlabeled set: %i = %1.2f"
            % (len_train_unlabeled, len_train_unlabeled / len_total)
        )

    def move_labeled_queries(self, X_query, Y_query, query_indices):
        # move new queries from unlabeled to labeled dataset
        self.X_train_labeled = self.X_train_labeled.append(X_query)
        self.X_train_unlabeled = self.X_train_unlabeled.drop(query_indices)

        try:
            self.Y_train_strong_labels = self.Y_train_strong_labels.append(
                self.Y_train_unlabeled.loc[query_indices]
            )
        except KeyError:
            # in a non experiment setting an error will be thrown because self.Y_train_unlabeled of course doesn't contains the labels
            for query_index in query_indices:
                self.Y_train_strong_labels.loc[query_index] = [-1]

        self.Y_train_labeled = self.Y_train_labeled.append(Y_query)
        self.Y_train_unlabeled = self.Y_train_unlabeled.drop(
            query_indices, errors="ignore"
        )

        # remove indices from all clusters in unlabeled and add to labeled
        for cluster_id in self.X_train_unlabeled_cluster_indices.keys():
            list_to_be_removed_and_appended = []
            for indice in query_indices:
                if indice in self.X_train_unlabeled_cluster_indices[cluster_id]:
                    list_to_be_removed_and_appended.append(indice)

            # don't change a list you're iterating over!
            for indice in list_to_be_removed_and_appended:
                self.X_train_unlabeled_cluster_indices[cluster_id].remove(indice)
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
