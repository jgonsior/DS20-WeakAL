import random

from activeLearner import ActiveLearner


class RandomSampler(ActiveLearner):
    def calculate_next_query_indices(self, X_train_unlabeled_cluster_indices):
        size_of_random_sample = self.data_storage.nr_queries_per_iteration
        length_X_train_unlabled = len(X_train_unlabeled_cluster_indices)
        if size_of_random_sample > length_X_train_unlabled:
            size_of_random_sample = length_X_train_unlabled
        return random.sample(X_train_unlabeled_cluster_indices,
                             size_of_random_sample)
