import random
from collections import defaultdict
from itertools import islice
from pprint import pprint

import numpy as np
from scipy.stats import entropy

from cluster_strategies import BaseClusterStrategy


class MostUncertainClusterStrategy(BaseClusterStrategy):
    def set_uncertainty_strategy(self, strategy):
        self.strategy = strategy

    def _get_random_cluster(self):
        # randomly select cluster
        random_cluster = random.choice(
            list(self.data_storage.X_train_unlabeled_cluster_indices.keys()))

        #  for cluster, cluster_indices in self.data_storage.X_train_unlabeled_cluster_indices.items(
        #  ):
        #  print(cluster, ":\t", len(cluster_indices))
        return random_cluster

    def get_cluster_indices(self, clf):
        # rank all clusters based on average k-most uncertainty
        k = self.data_storage.nr_queries_per_iteration

        highest_cumulative_uncertainty = 0
        highest_cumulative_uncertainty_cluster_id = None
        highest_cumulative_uncertainty_cluster_indices = None
        for cluster_id, cluster_indices in self.data_storage.X_train_unlabeled_cluster_indices.items(
        ):
            # calculate most uncertainty per
            Y_temp_proba = clf.predict_proba(
                self.data_storage.X_train_unlabeled.loc[cluster_indices])

            if self.strategy == 'least_confident':
                uncertainties = 1 - np.amax(Y_temp_proba, axis=1)
            elif self.strategy == 'max_margin':
                margin = np.partition(-Y_temp_proba, 1, axis=1)
                uncertainties = -np.abs(margin[:, 0] - margin[:, 1])
            elif self.strategy == 'entropy':
                uncertainties = np.apply_along_axis(entropy, 1, Y_temp_proba)

            # sum up top k uncertainties
            argsort = np.argsort(-uncertainties)[:k]
            top_k_cluster_indices = np.array(cluster_indices)[argsort]
            cumulative_uncertainty = np.sum(
                np.array(uncertainties)[argsort][:k])

            # length normalisation
            cumulative_uncertainty = cumulative_uncertainty / len(argsort[:k])

            if cumulative_uncertainty > highest_cumulative_uncertainty:
                highest_cumulative_uncertainty = cumulative_uncertainty
                highest_cumulative_uncertainty_cluster_id = cluster_id
                highest_cumulative_uncertainty_cluster_indices = top_k_cluster_indices.tolist(
                )

        return {
            highest_cumulative_uncertainty_cluster_id:
            highest_cumulative_uncertainty_cluster_indices
        }
