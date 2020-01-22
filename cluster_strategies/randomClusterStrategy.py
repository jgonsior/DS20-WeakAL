from itertools import islice
from pprint import pprint
import random
from collections import defaultdict

import numpy as np

from cluster_strategies import BaseClusterStrategy


class RandomClusterStrategy(BaseClusterStrategy):
    def _get_random_cluster(self):
        # randomly select cluster
        random_cluster = random.sample(range(0,
                                             self.cluster_model.n_clusters_),
                                       k=1)[0]
        for cluster, cluster_indices in self.data_storage.X_train_unlabeled_cluster_indices.items(
        ):
            print(cluster, ":\t", len(cluster_indices))
        return random_cluster

    def get_cluster_indices(self):
        random_cluster = self._get_random_cluster()
        print("Randomly selected cluster ", random_cluster)
        k = self.data_storage.nr_queries_per_iteration
        if k > len(self.data_storage.
                   X_train_unlabeled_cluster_indices[random_cluster]):
            return self.data_storage.X_train_unlabeled_cluster_indices[
                random_cluster]

        random_indices = random.sample(
            self.data_storage.
            X_train_unlabeled_cluster_indices[random_cluster],
            k=k)

        return {random_cluster: random_indices}
