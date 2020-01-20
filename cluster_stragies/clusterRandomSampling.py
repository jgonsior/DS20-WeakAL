from itertools import islice
from pprint import pprint
import random
from collections import defaultdict

import numpy as np

from active_learning_strategies import ClusterSampling


class RandomClusterSampling(ClusterSampling):
    def calculate_next_query_indices(self):
        random_cluster = self.get_random_cluster()
        k = self.nr_queries_per_iteration
        if k > len(self.X_train_unlabeled_clustered[random_cluster]):
            return self.X_train_unlabeled_clustered[random_cluster]

        random_indices = random.sample(
            self.X_train_unlabeled_clustered[random_cluster], k=k)

        return random_indices
