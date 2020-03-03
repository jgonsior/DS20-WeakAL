import argparse
import contextlib
import datetime
import hashlib
import io
import json
import logging
import multiprocessing
import os
import pickle
import random
import sys
from itertools import chain, combinations
from timeit import default_timer as timer

import numpy as np
#  import np.random.distributions as dists
import numpy.random
import pandas as pd
import peewee
import scipy
import sklearn.metrics
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from json_tricks import dumps
from playhouse.postgres_ext import *
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.metrics import (classification_report, confusion_matrix,
                             make_scorer, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     RandomizedSearchCV, ShuffleSplit,
                                     train_test_split)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler

from cluster_strategies import (DummyClusterStrategy,
                                MostUncertainClusterStrategy,
                                RandomClusterStrategy,
                                RoundRobinClusterStrategy)
from dataStorage import DataStorage
from experiment_setup_lib import (ExperimentResult, Logger,
                                  classification_report_and_confusion_matrix,
                                  divide_data, get_db,
                                  load_and_prepare_X_and_Y, standard_config,
                                  store_pickle, store_result)
from sampling_strategies import (BoundaryPairSampler, CommitteeSampler,
                                 RandomSampler, UncertaintySampler)


def train_al(X_train, Y_train, X_test, Y_test, label_encoder,
             hyper_parameters):
    hyper_parameters.len_train_data = len(Y_train)
    dataset_storage = DataStorage(hyper_parameters.random_seed)
    dataset_storage.set_training_data(X_train, Y_train, label_encoder,
                                      hyper_parameters.test_fraction,
                                      hyper_parameters.start_set_size, X_test,
                                      Y_test)

    if hyper_parameters.cluster == 'dummy':
        cluster_strategy = DummyClusterStrategy()
    elif hyper_parameters.cluster == 'random':
        cluster_strategy = RandomClusterStrategy()
    elif hyper_parameters.cluster == "MostUncertain_lc":
        cluster_strategy = MostUncertainClusterStrategy()
        cluster_strategy.set_uncertainty_strategy('least_confident')
    elif hyper_parameters.cluster == "MostUncertain_max_margin":
        cluster_strategy = MostUncertainClusterStrategy()
        cluster_strategy.set_uncertainty_strategy('max_margin')
    elif hyper_parameters.cluster == "MostUncertain_entropy":
        cluster_strategy = MostUncertainClusterStrategy()
        cluster_strategy.set_uncertainty_strategy('entropy')
    elif hyper_parameters.cluster == 'RoundRobin':
        cluster_strategy = RoundRobinClusterStrategy()

    cluster_strategy.set_data_storage(dataset_storage)

    active_learner_params = {
        'dataset_storage': dataset_storage,
        'cluster_strategy': cluster_strategy,
        'cores': hyper_parameters.cores,
        'random_seed': hyper_parameters.random_seed,
        'nr_learning_iterations': hyper_parameters.nr_learning_iterations,
        'nr_queries_per_iteration': hyper_parameters.nr_queries_per_iteration,
        'with_test': True,
    }

    if hyper_parameters.sampling == 'random':
        active_learner = RandomSampler(**active_learner_params)
    elif hyper_parameters.sampling == 'boundary':
        active_learner = BoundaryPairSampler(**active_learner_params)
    elif hyper_parameters.sampling == 'uncertainty_lc':
        active_learner = UncertaintySampler(**active_learner_params)
        active_learner.set_uncertainty_strategy('least_confident')
    elif hyper_parameters.sampling == 'uncertainty_max_margin':
        active_learner = UncertaintySampler(**active_learner_params)
        active_learner.set_uncertainty_strategy('max_margin')
    elif hyper_parameters.sampling == 'uncertainty_entropy':
        active_learner = UncertaintySampler(**active_learner_params)
        active_learner.set_uncertainty_strategy('entropy')
    #  elif hyper_parameters.sampling == 'committee':
    #  active_learner = CommitteeSampler(hyper_parameters.random_seed, hyper_parameters.cores, hyper_parameters.nr_learning_iterations)
    else:
        logging.error("No Active Learning Strategy specified")

    start = timer()
    trained_active_clf_list, metrics_per_al_cycle = active_learner.learn(
        minimum_test_accuracy_before_recommendations=hyper_parameters.
        minimum_test_accuracy_before_recommendations,
        with_cluster_recommendation=hyper_parameters.
        with_cluster_recommendation,
        with_uncertainty_recommendation=hyper_parameters.
        with_uncertainty_recommendation,
        with_snuba_lite=hyper_parameters.with_snuba_lite,
        cluster_recommendation_minimum_cluster_unity_size=hyper_parameters.
        cluster_recommendation_minimum_cluster_unity_size,
        cluster_recommendation_minimum_ratio_labeled_unlabeled=hyper_parameters
        .cluster_recommendation_ratio_labeled_unlabeled,
        uncertainty_recommendation_certainty_threshold=hyper_parameters.
        uncertainty_recommendation_certainty_threshold,
        uncertainty_recommendation_ratio=hyper_parameters.
        uncertainty_recommendation_ratio,
        snuba_lite_minimum_heuristic_accuracy=hyper_parameters.
        snuba_lite_minimum_heuristic_accuracy,
        stopping_criteria_uncertainty=hyper_parameters.
        stopping_criteria_uncertainty,
        stopping_criteria_acc=hyper_parameters.stopping_criteria_acc,
        stopping_criteria_std=hyper_parameters.stopping_criteria_std,
        allow_recommendations_after_stop=hyper_parameters.
        allow_recommendations_after_stop)
    end = timer()

    return trained_active_clf_list, end - start, metrics_per_al_cycle, dataset_storage, active_learner


def eval_al(X_test, Y_test, label_encoder, trained_active_clf_list, fit_time,
            metrics_per_al_cycle, param_distribution, dataset_storage,
            active_learner, hyper_parameters, dataset_path):
    hyper_parameters.amount_of_user_asked_queries = active_learner.amount_of_user_asked_queries

    classification_report_and_confusion_matrix_test = classification_report_and_confusion_matrix(
        trained_active_clf_list[0], X_test, Y_test,
        dataset_storage.label_encoder)
    classification_report_and_confusion_matrix_train = classification_report_and_confusion_matrix(
        trained_active_clf_list[0], dataset_storage.X_train_labeled,
        dataset_storage.Y_train_labeled, dataset_storage.label_encoder)

    # normalize by start_set_size
    percentage_user_asked_queries = 1 - hyper_parameters.amount_of_user_asked_queries / hyper_parameters.len_train_data
    test_acc = classification_report_and_confusion_matrix_test[0]['accuracy']

    if len(label_encoder.classes_) > 2:
        Y_scores = np.array(trained_active_clf_list[0].predict_proba(X_test))
        Y_test = Y_test.to_numpy().reshape(1, len(Y_scores))[0].tolist()

        roc_auc = roc_auc_score(
            Y_test,
            Y_scores,
            multi_class='ovo',
            average='macro',
            labels=[i for i in range(len(label_encoder.classes_))])
    else:
        Y_scores = trained_active_clf_list[0].predict_proba(X_test)[:, 1]
        #  print(Y_test.shape)
        Y_test = Y_test.to_numpy().reshape(1, len(Y_scores))[0].tolist()
        roc_auc = roc_auc_score(Y_test, Y_scores)
    # score is harmonic mean
    score = 2 * percentage_user_asked_queries * test_acc / (
        percentage_user_asked_queries + test_acc)

    # calculate based on params a unique id which should be the same across all similar cross validation splits
    unique_params = ""

    for k in param_distribution.keys():
        unique_params += str(vars(hyper_parameters)[k])

    param_list_id = hashlib.md5(unique_params.encode('utf-8')).hexdigest()

    db = get_db(db_name_or_type=hyper_parameters.db_name_or_type)
    params = hyper_parameters.get_params()
    params['dataset_path'] = dataset_path
    experiment_result = ExperimentResult(
        **params,
        amount_of_user_asked_queries=hyper_parameters.
        amount_of_user_asked_queries,
        metrics_per_al_cycle=dumps(metrics_per_al_cycle, allow_nan=True),
        fit_time=str(fit_time),
        confusion_matrix_test=dumps(
            classification_report_and_confusion_matrix_test[1],
            allow_nan=True),
        confusion_matrix_train=dumps(
            classification_report_and_confusion_matrix_train[1],
            allow_nan=True),
        classification_report_test=dumps(
            classification_report_and_confusion_matrix_test[0],
            allow_nan=True),
        classification_report_train=dumps(
            classification_report_and_confusion_matrix_train[0],
            allow_nan=True),
        acc_train=classification_report_and_confusion_matrix_train[0]
        ['accuracy'],
        acc_test=classification_report_and_confusion_matrix_test[0]
        ['accuracy'],
        fit_score=score,
        roc_auc=roc_auc,
        param_list_id=param_list_id)
    experiment_result.save()
    db.close()
    return score


'''
Takes a dataset_path, X, Y, label_encoder and does the following steps:
1. Split data
2. Train AL on the train dataset
3. Evaluate AL on the test dataset
4. Returns fit_score
'''


def train_and_eval_dataset(dataset_path, X_train, X_test, Y_train, Y_test,
                           label_encoder_classes, hyper_parameters,
                           param_distribution):
    label_encoder = LabelEncoder()
    label_encoder.fit(label_encoder_classes)

    trained_active_clf_list, fit_time, metrics_per_al_cycle, dataStorage, active_learner = train_al(
        X_train, Y_train, X_test, Y_test, label_encoder, hyper_parameters)

    fit_score = eval_al(X_test, Y_test, label_encoder, trained_active_clf_list,
                        fit_time, metrics_per_al_cycle, param_distribution,
                        dataStorage, active_learner, hyper_parameters,
                        dataset_path)
    return fit_score
