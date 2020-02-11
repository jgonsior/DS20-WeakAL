from itertools import islice
from pprint import pprint
import random
from collections import defaultdict

import numpy as np

from cluster_strategies import BaseClusterStrategy


class MostUncertainClusterStrategy(BaseClusterStrategy):
    def get_random_cluster(self):
        self.Y_train_unlabeled_cluster = self.cluster_model.fit_predict(
            self.X_train_unlabeled)

        self.X_train_unlabeled_clustered = defaultdict(lambda: list())
        for index, Y in enumerate(self.Y_train_unlabeled_cluster):
            self.X_train_unlabeled_clustered[Y].append(index)

        # randomly select cluster
        random_cluster = random.sample(range(0,
                                             self.cluster_model.n_clusters_),
                                       k=1)[0]
        return random_cluster

    def get_oracle_cluster(self):
        random_cluster = self.get_random_cluster()
        k = self.nr_queries_per_iteration
        if k > len(self.X_train_unlabeled_clustered[random_cluster]):
            return self.X_train_unlabeled_clustered[random_cluster]

        random_indices = random.sample(
            self.X_train_unlabeled_clustered[random_cluster], k=k)

        return random_indices

    def get_global_query_indice(self, cluster_query_indices):
        # return global_query_indices
        pass
