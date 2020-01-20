import random
from collections import defaultdict
from itertools import islice
from pprint import pprint

import numpy as np

from sampling_strategies import ClusterSampling


class OneClusterAtATimeSampling(ClusterSampling):
    def calculate_next_query_indices(self):
        random_cluster = self.get_random_cluster()

        k = self.nr_queries_per_iteration
        if k > len(self.X_train_unlabeled_clustered[random_cluster]):
            return self.X_train_unlabeled_clustered[random_cluster]

        random_indices = random.sample(
            X_train_unlabeled_clustered[random_cluster], k=k)

        return random_indices
