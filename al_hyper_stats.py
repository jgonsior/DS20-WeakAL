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

import numpy as np
import pandas as pd
import peewee
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from json_tricks import dumps
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     RandomizedSearchCV, train_test_split)
from sklearn.preprocessing import LabelEncoder

from cluster_strategies import (DummyClusterStrategy,
                                MostUncertainClusterStrategy,
                                RandomClusterStrategy,
                                RoundRobinClusterStrategy)
from dataStorage import DataStorage
from experiment_setup_lib import (Logger,
                                  classification_report_and_confusion_matrix,
                                  load_and_prepare_X_and_Y, standard_config,
                                  store_pickle, store_result, init_logging,
                                  get_db, ExperimentResult)
from sampling_strategies import (BoundaryPairSampler, CommitteeSampler,
                                 RandomSampler, UncertaintySampler)

standard_config = standard_config([])

init_logging(standard_config.output_dir)
db = get_db()

best_result = ExperimentResult.select().order_by(
    ExperimentResult.fit_score.desc()).limit(10)

for result in best_result:
    print("{:6,d} {:6.2%} {:6.2%}".format(result.amount_of_user_asked_queries,
                                          result.acc_test, result.fit_score))
