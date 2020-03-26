import argparse
import contextlib
import io
import os
import random
import sys

import numpy as np
from sklearn.model_selection import train_test_split

from al_cycle_wrapper import train_and_eval_dataset
from cluster_strategies import (
    DummyClusterStrategy,
    MostUncertainClusterStrategy,
    RandomClusterStrategy,
    RoundRobinClusterStrategy,
)
from dataStorage import DataStorage
from experiment_setup_lib import (
    Logger,
    classification_report_and_confusion_matrix,
    get_dataset,
    load_and_prepare_X_and_Y,
    standard_config,
)
from sampling_strategies import (
    BoundaryPairSampler,
    CommitteeSampler,
    RandomSampler,
    UncertaintySampler,
)

config = standard_config(
    [
        (
            ["--sampling"],
            {
                "required": True,
                "help": "Possible values: uncertainty, random, committe, boundary",
            },
        ),
        (["--dataset_name"], {"required": True,}),
        (
            ["--cluster"],
            {
                "default": "dummy",
                "help": "Possible values: dummy, random, mostUncertain, roundRobin",
            },
        ),
        (["--nr_learning_iterations"], {"type": int, "default": 150000}),
        (["--nr_queries_per_iteration"], {"type": int, "default": 150}),
        (["--start_set_size"], {"type": int, "default": 1}),
        (
            ["--minimum_test_accuracy_before_recommendations"],
            {"type": float, "default": 0.7},
        ),
        (
            ["--uncertainty_recommendation_certainty_threshold"],
            {"type": float, "default": 0.9},
        ),
        (["--uncertainty_recommendation_ratio"], {"type": float, "default": 1 / 100}),
        (["--snuba_lite_minimum_heuristic_accuracy"], {"type": float, "default": 0.9}),
        (
            ["--cluster_recommendation_minimum_cluster_unity_size"],
            {"type": float, "default": 0.7},
        ),
        (
            ["--cluster_recommendation_ratio_labeled_unlabeled"],
            {"type": float, "default": 0.9},
        ),
        (["--with_uncertainty_recommendation"], {"action": "store_true"}),
        (["--with_cluster_recommendation"], {"action": "store_true"}),
        (["--with_snuba_lite"], {"action": "store_true"}),
        (["--plot"], {"action": "store_true"}),
        (["--stopping_criteria_uncertainty"], {"type": float, "default": 0.7}),
        (["--stopping_criteria_acc"], {"type": float, "default": 0.7}),
        (["--stopping_criteria_std"], {"type": float, "default": 0.7}),
        (
            ["--allow_recommendations_after_stop"],
            {"action": "store_true", "default": False},
        ),
        (["--db_name_or_type"], {"default": "sqlite"}),
        (["--hyper_search_type"], {"default": "random"}),
        (["--user_query_budget_limit"], {"type": float, "default": 200}),
    ]
)

X_train, X_test, Y_train, Y_test, label_encoder_classes = get_dataset(
    config.datasets_path, config.dataset_name
)

score = train_and_eval_dataset(
    config.dataset_name,
    X_train,
    X_test,
    Y_train,
    Y_test,
    label_encoder_classes,
    hyper_parameters=vars(config),
)
print("Done with ", score)
