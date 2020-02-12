from itertools import islice
from pprint import pprint
import random
from collections import defaultdict

import numpy as np

from cluster_strategies import BaseClusterStrategy


class DummyClusterStrategy(BaseClusterStrategy):
    def get_cluster_indices(self, **kwargs):
        return self.data_storage.X_train_unlabeled_cluster_indices
