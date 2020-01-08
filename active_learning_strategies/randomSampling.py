import random

from activeLearner import ActiveLearner


class RandomSampler(ActiveLearner):
    def retrieve_query_indices(self):
        return random.sample(range(len(self.X_query)),
                             self.nr_queries_per_iteration)
