import math
import argparse
import contextlib
import datetime
import io
import logging
import multiprocessing
import os
import random
import sys
from itertools import chain, combinations
from timeit import default_timer as timer

import altair as alt
import altair_viewer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peewee
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from json_tricks import dumps, loads
from playhouse.migrate import *
from playhouse.postgres_ext import *
from playhouse.shortcuts import model_to_dict
from scipy.stats import randint, uniform
from sklearn.datasets import load_iris
from tabulate import tabulate

from cluster_strategies import (
    DummyClusterStrategy,
    MostUncertainClusterStrategy,
    RandomClusterStrategy,
    RoundRobinClusterStrategy,
)
from dataStorage import DataStorage
from experiment_setup_lib import (
    BaseModel,
    ExperimentResult,
    classification_report_and_confusion_matrix,
    get_db,
    get_single_al_run_stats_row,
    get_single_al_run_stats_table_header,
    load_and_prepare_X_and_Y,
    standard_config,
    calculate_global_score,
)
from sampling_strategies import (
    BoundaryPairSampler,
    CommitteeSampler,
    RandomSampler,
    UncertaintySampler,
)

alt.renderers.enable("altair_viewer")
#  alt.renderers.enable('vegascope')

config = standard_config([(["--db"], {"default": "sqlite"}),])

db = get_db(db_name_or_type=config.db)

db.execute_sql("DROP TABLE experimentresult;")
db.create_tables([ExperimentResult])
db.execute_sql("INSERT INTO experimentresult (SELECT * FROM experimentresult_old);")

global_score_no_weak_roc_auc_field = peewee.FloatField(index=True, null=True)
global_score_no_weak_acc_field = peewee.FloatField(index=True, null=True)
global_score_with_weak_roc_auc_field = peewee.FloatField(index=True, null=True)
global_score_with_weak_acc_field = peewee.FloatField(index=True, null=True)

global_score_no_weak_roc_auc_norm_field = peewee.FloatField(index=True, null=True)
global_score_no_weak_acc_norm_field = peewee.FloatField(index=True, null=True)
global_score_with_weak_roc_auc_norm_field = peewee.FloatField(index=True, null=True)
global_score_with_weak_acc_norm_field = peewee.FloatField(index=True, null=True)


migrator = PostgresqlMigrator(db)

migrate(
    migrator.add_column(
        "experimentresult",
        "global_score_no_weak_roc_auc",
        global_score_no_weak_roc_auc_field,
    ),
    migrator.add_column(
        "experimentresult", "global_score_no_weak_acc", global_score_no_weak_acc_field,
    ),
    migrator.add_column(
        "experimentresult",
        "global_score_with_weak_roc_auc",
        global_score_with_weak_roc_auc_field,
    ),
    migrator.add_column(
        "experimentresult",
        "global_score_with_weak_acc",
        global_score_with_weak_acc_field,
    ),
    migrator.add_column(
        "experimentresult",
        "global_score_no_weak_roc_auc_norm",
        global_score_no_weak_roc_auc_norm_field,
    ),
    migrator.add_column(
        "experimentresult",
        "global_score_no_weak_acc_norm",
        global_score_no_weak_acc_norm_field,
    ),
    migrator.add_column(
        "experimentresult",
        "global_score_with_weak_roc_auc_norm",
        global_score_with_weak_roc_auc_norm_field,
    ),
    migrator.add_column(
        "experimentresult",
        "global_score_with_weak_acc_norm",
        global_score_with_weak_acc_norm_field,
    ),
    migrator.rename_column(
        "experimentresult", "global_score", "global_score_with_weak_roc_auc_old"
    ),
    migrator.rename_column(
        "experimentresult",
        "global_score_norm",
        "global_score_with_weak_roc_auc_norm_old",
    ),
)

# @todo: include here the OLD definition instead!
class ExperimentResult(BaseModel):
    id_field = peewee.AutoField()

    # hyper params
    datasets_path = peewee.TextField()
    dataset_name = peewee.TextField()
    db_name_or_type = peewee.TextField()
    classifier = peewee.TextField(index=True)
    cores = peewee.IntegerField()
    test_fraction = peewee.FloatField()
    sampling = peewee.TextField(index=True)
    random_seed = peewee.IntegerField()
    cluster = peewee.TextField(index=True)
    nr_learning_iterations = peewee.IntegerField()
    nr_queries_per_iteration = peewee.IntegerField(index=True)
    start_set_size = peewee.FloatField(index=True)
    with_uncertainty_recommendation = peewee.BooleanField(index=True)
    with_cluster_recommendation = peewee.BooleanField(index=True)
    with_snuba_lite = peewee.BooleanField(index=True)
    uncertainty_recommendation_certainty_threshold = peewee.FloatField(null=True)
    uncertainty_recommendation_ratio = peewee.FloatField(null=True)
    snuba_lite_minimum_heuristic_accuracy = peewee.FloatField(null=True)
    cluster_recommendation_minimum_cluster_unity_size = peewee.FloatField(null=True)
    cluster_recommendation_ratio_labeled_unlabeled = peewee.FloatField(null=True)
    metrics_per_al_cycle = BinaryJSONField()  # json string
    amount_of_user_asked_queries = peewee.IntegerField(index=True)
    allow_recommendations_after_stop = peewee.BooleanField()
    stopping_criteria_uncertainty = peewee.FloatField()
    stopping_criteria_acc = peewee.FloatField()
    stopping_criteria_std = peewee.FloatField()

    # information of hyperparam run
    experiment_run_date = peewee.DateTimeField(default=datetime.datetime.now)
    fit_time = peewee.TextField()  # timedelta
    confusion_matrix_test = BinaryJSONField()  # json
    confusion_matrix_train = BinaryJSONField()  # json
    classification_report_train = BinaryJSONField()  # json
    classification_report_test = BinaryJSONField()  # json
    acc_train = peewee.FloatField(index=True)
    acc_test = peewee.FloatField(index=True)
    fit_score = peewee.FloatField(index=True)
    roc_auc = peewee.FloatField(index=True)
    global_score_with_weak_roc_auc_old = peewee.FloatField(index=True)
    global_score_with_weak_roc_auc_norm_old = peewee.FloatField(index=True)

    global_score_no_weak_roc_auc = peewee.FloatField(index=True, null=True)
    global_score_no_weak_acc = peewee.FloatField(index=True, null=True)
    global_score_with_weak_roc_auc = peewee.FloatField(index=True, null=True)
    global_score_with_weak_acc = peewee.FloatField(index=True, null=True)

    global_score_no_weak_roc_auc_norm = peewee.FloatField(index=True, null=True)
    global_score_no_weak_acc_norm = peewee.FloatField(index=True, null=True)
    global_score_with_weak_roc_auc_norm = peewee.FloatField(index=True, null=True)
    global_score_with_weak_acc_norm = peewee.FloatField(index=True, null=True)

    param_list_id = peewee.TextField(index=True)

    cv_fit_score_mean = peewee.FloatField(null=True)
    cv_fit_score_std = peewee.FloatField(null=True)

    thread_id = peewee.BigIntegerField(index=True)
    end_time = peewee.DateTimeField(index=True)


db.bind([ExperimentResult])

# calculate new roc_auc scores
# random is amount of 1/amount_of_labels, not fixed 0.5!!

for experimentresult in ExperimentResult.select(ExperimentResult):
    metrics_per_al_cycle = loads(experimentresult.metrics_per_al_cycle)
    amount_of_labels = 5

    amount_of_labels_per_metric_value = metrics_per_al_cycle["query_length"]

    amount_of_labels_per_metric_value_norm = [
        math.log2(m) for m in amount_of_labels_per_metric_value
    ]

    metric_values = metrics_per_al_cycle["all_unlabeled_roc_auc_scores"]
    experimentresult.global_score_no_weak_roc_auc = calculate_global_score(
        metric_values, amount_of_labels_per_metric_value, amount_of_labels
    )
    experimentresult.global_score_no_weak_roc_auc_norm = calculate_global_score(
        metric_values, amount_of_labels_per_metric_value_norm, amount_of_labels
    )
    experimentresult.global_score_no_weak_acc = calculate_global_score(
        metric_values, amount_of_labels_per_metric_value, amount_of_labels
    )
    experimentresult.global_score_no_weak_acc_norm = calculate_global_score(
        metric_values, amount_of_labels_per_metric_value_norm, amount_of_labels
    )

    experimentresult.global_score_with_weak_roc_auc = calculate_global_score(
        metric_values, amount_of_labels_per_metric_value, amount_of_labels
    )
    experimentresult.global_score_with_weak_roc_auc_norm = calculate_global_score(
        metric_values, amount_of_labels_per_metric_value_norm, amount_of_labels
    )
    experimentresult.global_score_with_weak_acc = calculate_global_score(
        metric_values, amount_of_labels_per_metric_value, amount_of_labels
    )
    experimentresult.global_score_with_weak_acc_norm = calculate_global_score(
        metric_values, amount_of_labels_per_metric_value_norm, amount_of_labels
    )

    print(experimentresult.id_field)
    experimentresult.save()


migrate(
    migrator.add_not_null("experimentresult", "global_score_no_weak_roc_auc"),
    migrator.add_not_null("experimentresult", "global_score_no_weak_acc"),
    migrator.add_not_null("experimentresult", "global_score_with_weak_roc_auc"),
    migrator.add_not_null("experimentresult", "global_score_with_weak_acc"),
    migrator.add_not_null("experimentresult", "global_score_no_weak_roc_auc_norm"),
    migrator.add_not_null("experimentresult", "global_score_no_weak_acc_norm"),
    migrator.add_not_null("experimentresult", "global_score_with_weak_roc_auc_norm"),
    migrator.add_not_null("experimentresult", "global_score_with_weak_acc_norm"),
)
