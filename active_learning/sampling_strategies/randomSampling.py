import random

from ..activeLearner import ActiveLearner


class RandomSampler(ActiveLearner):
    def calculate_next_query_indices(self, X_train_unlabeled_cluster_indices):
        # we only take the first cluster into consideration and take a random sample from it, could be extended to work with multiple clusters as well
        random_cluster = X_train_unlabeled_cluster_indices[
            list(X_train_unlabeled_cluster_indices.keys())[0]
        ]

        size_of_random_sample = self.nr_queries_per_iteration
        length_X_train_unlabled = len(random_cluster)
        if size_of_random_sample > length_X_train_unlabled:
            size_of_random_sample = length_X_train_unlabled

        return random.sample(random_cluster, size_of_random_sample)
