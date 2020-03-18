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
from playhouse.migrate import *
import altair as alt
import altair_viewer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peewee
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from json_tricks import dumps, loads
from playhouse.shortcuts import model_to_dict
from scipy.stats import randint, uniform
from sklearn.datasets import load_iris
from tabulate import tabulate
from playhouse.migrate import *
from cluster_strategies import (
    DummyClusterStrategy,
    MostUncertainClusterStrategy,
    RandomClusterStrategy,
    RoundRobinClusterStrategy,
)
from dataStorage import DataStorage
from experiment_setup_lib import (
    ExperimentResult,
    classification_report_and_confusion_matrix,
    get_db,
    get_single_al_run_stats_row,
    get_single_al_run_stats_table_header,
    load_and_prepare_X_and_Y,
    standard_config,
)
from sampling_strategies import (
    BoundaryPairSampler,
    CommitteeSampler,
    RandomSampler,
    UncertaintySampler,
)

alt.renderers.enable("altair_viewer")
#  alt.renderers.enable('vegascope')

config = standard_config(
    [
        (["--limit"], {"type": int, "default": 10}),
        (["--param_list_id"], {"default": "-1"}),
        (["--id"], {"type": int, "default": -1,}),
        (["--db"], {"default": "sqlite"}),
        (["--dataset_stats"], {"action": "store_true"}),
        (["--param_list_stats"], {"action": "store_true"}),
    ]
)

db = get_db(db_name_or_type=config.db)

global_score_no_weak_roc_auc_field = peewee.FloatField(index=True)
global_score_no_weak_acc_field = peewee.FloatField(index=True)

migrator = PostgresqlMigrator(db)
migrate(migrator.add_column("experimentresult", "global_score_no_weak_roc_auc",))
