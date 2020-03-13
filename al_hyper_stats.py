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
from playhouse.shortcuts import model_to_dict
from scipy.stats import randint, uniform
from sklearn.datasets import load_iris
from tabulate import tabulate

from cluster_strategies import (DummyClusterStrategy,
                                MostUncertainClusterStrategy,
                                RandomClusterStrategy,
                                RoundRobinClusterStrategy)
from dataStorage import DataStorage
from experiment_setup_lib import (ExperimentResult,
                                  classification_report_and_confusion_matrix,
                                  get_db, get_single_al_run_stats_row,
                                  get_single_al_run_stats_table_header,
                                  load_and_prepare_X_and_Y, standard_config)
from sampling_strategies import (BoundaryPairSampler, CommitteeSampler,
                                 RandomSampler, UncertaintySampler)

alt.renderers.enable('altair_viewer')
#  alt.renderers.enable('vegascope')

config = standard_config([
    (['--limit'], {
        'type': int,
        'default': 10
    }),
    (['--param_list_id'], {
        'default': "-1"
    }),
    (['--id'], {
        'type': int,
        'default': -1
    }),
    (['--db'], {
        'default': 'sqlite'
    }),
    (['--dataset_stats'], {
        'action': 'store_true'
    }),
    (['--param_list_stats'], {
        'action': 'store_true'
    }),
])

#  id zu untersuchen: 31858014d685a3f1ba3e4e32690ddfc3 -> warum ist roc_auc dort so falsch?

db = get_db(db_name_or_type=config.db)

if config.dataset_stats:
    # select count(*), dataset_name from experimentresult group by dataset_name;
    results = ExperimentResult.select(
        ExperimentResult.dataset_name,
        peewee.fn.COUNT(
            ExperimentResult.id_field).alias('dataset_name_count')).group_by(
                ExperimentResult.dataset_name)

    for result in results:
        print("{:>4,d} {}".format(result.dataset_name_count,
                                  result.dataset_name))

elif config.param_list_stats:
    #  SELECT param_list_id, avg(fit_score), stddev(fit_score), avg(global_score), stddev(global_score), avg(start_set_size) as sss, count(*) FROM experimentresult WHERE start_set_size = 1 GROUP BY param_list_id ORDER BY 7 DESC, 4 DESC LIMIT 30;

    results = ExperimentResult.select(
        ExperimentResult.param_list_id,
        peewee.fn.AVG(ExperimentResult.fit_score).alias('avg_fit_score'),
        peewee.fn.STDDEV(ExperimentResult.fit_score).alias('stddev_fit_score'),
        peewee.fn.AVG(ExperimentResult.global_score).alias('avg_global_score'),
        peewee.fn.STDDEV(
            ExperimentResult.global_score).alias('stddev_global_score'),
        peewee.fn.AVG(ExperimentResult.start_set_size).alias('sss'),
        peewee.fn.COUNT(ExperimentResult.id_field).alias('count')).group_by(
            ExperimentResult.param_list_id).order_by(
                peewee.fn.COUNT(ExperimentResult.id_field).desc(),
                peewee.fn.AVG(ExperimentResult.fit_score).desc()).limit(20)

    table = []

    for result in results:
        data = vars(result)
        data['param_list_id'] = data['__data__']['param_list_id']
        del data['__data__']
        del data['_dirty']
        del data['__rel__']
        table.append(vars(result))

    print(tabulate(table, headers="keys"))

elif config.param_list_id != "-1":
    # SELECT id_field, param_list_id, dataset_path, start_set_size as sss, sampling, cluster, allow_recommendations_after_stop as SA, stopping_criteria_uncertainty as SCU, stopping_criteria_std as SCS, stopping_criteria_acc as SCA, amount_of_user_asked_queries as "#q", acc_test, fit_score, global_score_norm, thread_id, end_time from experimentresult where param_list_id='31858014d685a3f1ba3e4e32690ddfc3' order by end_time, fit_score desc, param_list_id;
    results = ExperimentResult.select().where(
        ExperimentResult.param_list_id == config.param_list_id)
    for result in results:

        metrics = loads(result.metrics_per_al_cycle)

        data = pd.DataFrame({
            'iteration':
            range(0, len(metrics['all_unlabeled_roc_auc_scores'])),
            'all_unlabeled_roc_auc_scores':
            metrics['all_unlabeled_roc_auc_scores'],
            'query_length':
            metrics['query_length'],
            'recommendation':
            metrics['recommendation'],
            'query_strong_accuracy_list':
            metrics['query_strong_accuracy_list'],
        })

        alt.Chart(data).mark_line(point=True, size=60).encode(
            x='iteration',
            y='all_unlabeled_roc_auc_scores',
            tooltip=[
                'query_strong_accuracy_list', 'query_length', 'recommendation'
            ]).interactive()
        chart.serve()

        amount_of_clusters = 0
        amount_of_certainties = 0
        amount_of_active = 0
        for recommendation, query_length in zip(metrics['recommendation'],
                                                metrics['query_length']):
            if recommendation == 'U':
                amount_of_certainties += 1
            if recommendation == 'C':
                amount_of_clusters += 1
            if recommendation == 'A':
                amount_of_active += 1
        #SELECT id_field, param_list_id, dataset_path, start_set_size as sss, sampling, cluster, allow_recommendations_after_stop as SA, stopping_criteria_uncertainty as SCU, stopping_criteria_std as SCS, stopping_criteria_acc as SCA, amount_of_user_asked_queries as "#q", acc_test, fit_score, global_score_norm, thread_id, end_time from experimentresult where param_list_id='31858014d685a3f1ba3e4e32690ddfc3' order by end_time, fit_score desc, param_list_id;
        print(
            "{:>20} {:6,d} {:>25} {:>25} {:4,d} {:4,d} {:4,d} {:3,d} {:5,d} {:6.2} {:6.2%} {:6.2%} {:6.2%} {:6.2%}"
            .format(
                result.param_list_id,
                result.id_field,
                result.sampling,
                result.cluster,
                amount_of_clusters,
                amount_of_certainties,
                amount_of_active,
                result.allow_recommendations_after_stop,
                result.amount_of_user_asked_queries,
                result.stopping_criteria_uncertainty,
                result.stopping_criteria_std,
                result.stopping_criteria_acc,
                result.acc_test,
                result.fit_score,
            ))

    exit(-3)

elif config.id != -1:
    print(get_single_al_run_stats_table_header())
    result = ExperimentResult.get(ExperimentResult.id_field == config.id)
    metrics = loads(result.metrics_per_al_cycle)

    for key in metrics.keys():
        print(len(metrics[key]), "\t", key)

    for i in range(0, len(metrics['recommendation']) - 1):
        print(get_single_al_run_stats_row(i, None, None, metrics, index=i))
else:
    best_result = (ExperimentResult.select(
        ExperimentResult.param_list_id,
        peewee.fn.AVG(ExperimentResult.param_list_id)).group_by(
            ExperimentResult.param_list_id).limit(config.limit))

    print(
        "{:>20} {:>6} {:>25} {:>25} {:>4} {:>4} {:>4} {:>3} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6}"
        .format("hash", "Id", "Sampling", "Cluster", "#C", "#U", "#A", "AS",
                "#q", "SU", "SS", "SA", "acc_te", "fit"))

    for result in best_result:
        metrics = loads(result.metrics_per_al_cycle)

        amount_of_clusters = 0
        amount_of_certainties = 0
        amount_of_active = 0
        for recommendation, query_length in zip(metrics['recommendation'],
                                                metrics['query_length']):
            if recommendation == 'U':
                amount_of_certainties += 1
            if recommendation == 'C':
                amount_of_clusters += 1
            if recommendation == 'A':
                amount_of_active += 1

        print(
            "{:>20} {:6,d} {:>25} {:>25} {:4,d} {:4,d} {:4,d} {:3,d} {:5,d} {:6.2} {:6.2%} {:6.2%} {:6.2%} {:6.2%}"
            .format(
                result.param_list_id,
                result.id_field,
                result.sampling,
                result.cluster,
                amount_of_clusters,
                amount_of_certainties,
                amount_of_active,
                result.allow_recommendations_after_stop,
                result.amount_of_user_asked_queries,
                result.stopping_criteria_uncertainty,
                result.stopping_criteria_std,
                result.stopping_criteria_acc,
                result.acc_test,
                result.fit_score,
            ))

# print for best run all the metrics from active_learner.py
