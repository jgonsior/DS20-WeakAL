import scipy
import abc
from collections import Counter, defaultdict
from math import e, log

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import (
    OPTICS,
    MiniBatchKMeans,
    cluster_optics_dbscan,
    AgglomerativeClustering,
)
import copy
import inspect
import multiprocessing
from collections import defaultdict

import pandas as pd
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit

from active_learning.al_cycle_wrapper import train_and_eval_dataset
from active_learning.experiment_setup_lib import (
    get_dataset,
    get_param_distribution,
    log_it,
    standard_config,
)
from active_learning.experiment_setup_lib import log_it, get_dataset, standard_config


standard_config = standard_config(
    [
        (["--NR_LEARNING_ITERATIONS"], {"type": int, "default": 1000000}),
        (["--CV"], {"type": int, "default": 3}),
        (["--NR_RANDOM_RUNS"], {"type": int, "default": 200000}),
        (["--POPULATION_SIZE"], {"type": int, "default": 100}),
        (["--TOURNAMENT_SIZE"], {"type": int, "default": 100}),
        (["--GENERATIONS_NUMBER"], {"type": int, "default": 100}),
        (["--GENE_MUTATION_PROB"], {"type": float, "default": 0.3}),
        (["--DB_NAME_OR_TYPE"], {"default": "sqlite"}),
        (["--HYPER_SEARCH_TYPE"], {"default": "random"}),
        (["--DATASET_NAME"], {}),
    ]
)

zero_to_one = scipy.stats.uniform(loc=0, scale=1)
half_to_one = scipy.stats.uniform(loc=0.5, scale=0.5)
#  nr_queries_per_iteration = scipy.stats.randint(1, 151)
NR_QUERIES_PER_ITERATION = [10]
#  START_SET_SIZE = scipy.stats.uniform(loc=0.001, scale=0.1)
#  START_SET_SIZE = [1, 10, 25, 50, 100]
START_SET_SIZE = [1]

param_distribution = {
    "n_cluster_dividend": scipy.stats.randint(1, 70),
    "cluster_algo": ["kmean", "aglo"],
}


class Estimator(BaseEstimator):
    def set_params(self, **params):
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        for k, v in params.items():
            setattr(self, k, v)

        return self

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X_train, Y_train, **kwargs):
        n_samples, n_features = X_train.shape
        if self.cluster_algo == "aglo":
            # then cluster it
            cluster_model = AgglomerativeClustering(
                n_clusters=int(n_samples / self.n_cluster_dividend)
            )  # distance_threshold=0,                                                     n_clusters=None)
        #  plot_cluster()
        #  plot_dendrogram()

        elif self.cluster_algo == "kmean":
            cluster_model = MiniBatchKMeans(
                n_clusters=int(n_samples / self.n_cluster_dividend),
                batch_size=min(int(n_samples / 100), int(n_features)),
            )

        #  cluster_model = OPTICS(min_cluster_size=20, n_jobs=n_jobs)
        #  with np.errstate(divide="ignore"):
        #  cluster_model.fit(X_train_unlabeled)

        Y_train_cluster = cluster_model.fit_predict(X_train)
        #  # fit cluster
        #  Y_train_cluster = cluster _model.labels_[
        #  cluster_model.ordering_
        #  ]

        # check the labels per cluster

        clusters = defaultdict(lambda: list())

        for cluster_index, Y in zip(Y_train_cluster, Y_train):
            clusters[cluster_index].append(Y)
        data = []
        for cluster_id, Y_cluster in clusters.items():
            counter = Counter(Y_cluster)
            if (
                counter.most_common(1)[0][1] / len(Y_cluster) > 0.9
                and len(Y_cluster) > 5
            ):
                data.append(
                    "{}: {} {}".format(
                        counter.most_common(1)[0][1] / len(Y_cluster),
                        counter.most_common(1)[0][0],
                        Y_cluster,
                    )
                )
        self.scor = len(data)

    def score(self, X, Y):
        return self.scor


active_learner = Estimator()
#  print(active_learner.get_params().keys())

grid = RandomizedSearchCV(
    active_learner,
    param_distribution,
    n_iter=standard_config.NR_RANDOM_RUNS,
    pre_dispatch=standard_config.N_JOBS,
    return_train_score=False,
    verbose=9999999999999999999999999999999999,
    # verbose=0,
    n_jobs=standard_config.N_JOBS,
    refit=False,
)
X_train, X_test, Y_train, Y_test, label_encoder_classes = get_dataset(
    standard_config.DATASETS_PATH, standard_config.DATASET_NAME
)
grid = grid.fit(X_train.to_numpy(), Y_train[0].to_list())

print(grid.best_params_)
print(grid.best_score_)
print(
    pd.DataFrame(grid.cv_results_)
    .sort_values("mean_test_score", ascending=False)
    .head()
)
