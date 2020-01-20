import random
from collections import defaultdict
from itertools import islice
from pprint import pprint

import numpy as np

from sampling_strategies import ClusterSampling


class MultipleClusterAtATimeSampling(ClusterSampling):
    def calculate_next_query_indices(self):
        Y_train_unlabeled_cluster = self.cluster_model.fit_predict(
            self.pca.transform(self.X_train_unlabeled))

        X_train_unlabeled_clustered = defaultdict(lambda: list())
        for index, Y in enumerate(Y_train_unlabeled_cluster):
            X_train_unlabeled_clustered[Y].append(index)

        # select n random clusters and then evenly random elements from them instead
        -> two cluster families, eine die von verschiedenen cluster daten holt, eine die daten nur pro cluster holt
        # randomly select cluster
        # randomly select indices from it
        random_cluster = random.sample(range(0,
                                             self.cluster_model.n_clusters_),
                                       k=1)[0]
        k = self.nr_queries_per_iteration
        if k > len(X_train_unlabeled_clustered[random_cluster]):
            return X_train_unlabeled_clustered[random_cluster]

        random_indices = random.sample(
            X_train_unlabeled_clustered[random_cluster], k=k)

        return random_indices
