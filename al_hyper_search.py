import numpy as np
import random
import sys
import hashlib
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
    init_logger,
    log_it,
    standard_config,
)
from fake_experiment_oracle import FakeExperimentOracle

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
    ]
)
init_logger(standard_config.LOG_FILE)
param_distribution = get_param_distribution(**vars(standard_config))


class Estimator(BaseEstimator):
    # dirty hack to allow kwargs in init :)
    def set_params(self, **params):
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        for k, v in params.items():
            setattr(self, k, v)

        return self

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, dataset_names, Y_not_used, **kwargs):
        init_logger(standard_config.LOG_FILE)
        if self.RANDOM_SEED == -2:
            self.RANDOM_SEED = random.randint(0, 2147483647)
            np.random.seed(self.RANDOM_SEED)
            random.seed(self.RANDOM_SEED)
            print(self.RANDOM_SEED)
        self.scores = []

        for dataset_name in dataset_names:
            if dataset_name is None:
                continue
            #  gc.collect()

            X_train, X_test, Y_train, Y_test, label_encoder_classes = get_dataset(
                standard_config.DATASETS_PATH, dataset_name, self.RANDOM_SEED
            )
            score, Y_train_al = train_and_eval_dataset(
                dataset_name,
                X_train,
                X_test,
                Y_train,
                Y_test,
                label_encoder_classes,
                hyper_parameters=vars(self),
                oracle=FakeExperimentOracle(),
            )

            self.scores.append(score)
            log_it(dataset_name + " done with " + str(self.scores[-1]))
            # gc.collect()

    def score(self, dataset_names_should_be_none, Y_not_used):
        #  print("score", dataset_names_should_be_none)
        return sum(self.scores) / len(self.scores)


active_learner = Estimator()
print(active_learner.get_params().keys())

if standard_config.NR_LEARNING_ITERATIONS == 3:
    X = ["dwtc", "ibn_sina"]
else:
    X = [
        "dwtc",
        "dwtc",
        "dwtc",
        #  "ibn_sina",
        #  "hiva",
        #  "orange",
        #  "sylva",
        #  'forest_covtype',
        #  "zebra",
    ]

X.append(None)
Y = [None] * len(X)


# fake our own stupid cv=1 split
class NoCvCvSplit:
    def __init__(self, n_splits=3, **kwargs):
        self.n_splits = n_splits
        pass

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

    def split(self, dataset_names, y=None, groups=None, split_quota=None):
        # everything goes into train once
        train_idx = [i for i in range(0, len(dataset_names) - 1)]

        yield train_idx, [len(dataset_names) - 1]


if standard_config.HYPER_SEARCH_TYPE == "random":
    grid = RandomizedSearchCV(
        active_learner,
        param_distribution,
        n_iter=standard_config.NR_RANDOM_RUNS,
        pre_dispatch=standard_config.N_JOBS,
        return_train_score=False,
        cv=NoCvCvSplit(n_splits=1),
        verbose=9999999999999999999999999999999999,
        # verbose=0,
        n_jobs=standard_config.N_JOBS,
        refit=False,
    )
    grid = grid.fit(X, Y)
elif standard_config.HYPER_SEARCH_TYPE == "evo":
    grid = EvolutionaryAlgorithmSearchCV(
        estimator=active_learner,
        params=param_distribution,
        verbose=True,
        cv=ShuffleSplit(test_size=0.20, n_splits=1, random_state=0),  # fake CV=1 split
        population_size=standard_config.POPULATION_SIZE,
        gene_mutation_prob=standard_config.GENE_MUTATION_PROB,
        tournament_size=standard_config.TOURNAMENT_SIZE,
        generations_number=standard_config.GENERATIONS_NUMBER,
        n_jobs=standard_config.N_JOBS,
    )
    grid.fit(X, Y)

print(grid.best_params_)
print(grid.best_score_)
print(
    pd.DataFrame(grid.cv_results_)
    .sort_values("mean_test_score", ascending=False)
    .head()
)
