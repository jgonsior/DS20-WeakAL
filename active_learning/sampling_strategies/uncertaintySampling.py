from itertools import chain

import numpy as np
from scipy.stats import entropy

from ..activeLearner import ActiveLearner


class UncertaintySampler(ActiveLearner):
    def set_uncertainty_strategy(self, strategy):
        self.strategy = strategy

    def setClassifierClasses(self, classes):
        self.classifier_classes = classes

    def calculate_next_query_indices(self, X_train_unlabeled_cluster_indices, *args):
        # merge indices from all clusters together and take the n most uncertain ones from them
        X_train_unlabeled_indices = list(
            chain(*list(X_train_unlabeled_cluster_indices.values()))
        )
        # recieve predictions and probabilitys
        # for all possible classifications of CLASSIFIER
        Y_temp_proba = self.clf_list[0].predict_proba(
            self.data_storage.X_train_unlabeled.loc[X_train_unlabeled_indices]
        )

        if self.strategy == "least_confident":
            result = 1 - np.amax(Y_temp_proba, axis=1)
        elif self.strategy == "max_margin":
            margin = np.partition(-Y_temp_proba, 1, axis=1)
            result = -np.abs(margin[:, 0] - margin[:, 1])
        elif self.strategy == "entropy":
            result = np.apply_along_axis(entropy, 1, Y_temp_proba)

        # sort X_train_unlabeled_indices by argsort
        argsort = np.argsort(-result)
        query_indices = np.array(X_train_unlabeled_indices)[argsort]

        # return smallest probabilities
        return query_indices[: self.nr_queries_per_iteration]
