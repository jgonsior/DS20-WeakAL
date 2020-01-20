import random

from activeLearner import ActiveLearner


class RandomSampler(ActiveLearner):
    def calculate_next_query_indices(self, X_train_unlabeled):
        return random.sample(range(len(X_train_unlabeled)),
                             self.nr_queries_per_iteration)
