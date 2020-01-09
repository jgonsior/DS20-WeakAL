from pprint import pprint
import random
from itertools import islice
from activeLearner import ActiveLearner
import numpy as np


class SheetBasedActiveLearner(ActiveLearner):
    current_sheet_name = None

    def __init__(self, config):
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

        super(SheetBasedActiveLearner, self).__init__(config)

    def move_new_queries_to_training_data(self, query_indices):
        print("Queried sheet " + self.current_sheet_name)
        self.metrics_per_al_cycle['queried_sheet'].append(self.current_sheet_name)
        super(SheetBasedActiveLearner,
              self).move_new_queries_to_training_data(query_indices)
        self.X_query_spreadsheets = np.delete(self.X_query_spreadsheets,
                                              query_indices, 0)
