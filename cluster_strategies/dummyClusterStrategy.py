from itertools import islice
from pprint import pprint
import random
from collections import defaultdict

import numpy as np

from cluster_strategies import BaseClusterStrategy


class DummyClusterStrategy(BaseClusterStrategy):
    def get_oracle_cluster(self):
        # return X_train_unlabeled
        pass

    def get_global_query_indice(self, cluster_query_indices):
        # return global_query_indices
        pass
