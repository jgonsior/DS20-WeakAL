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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler

db = peewee.DatabaseProxy()


class BaseModel(peewee.Model):
    class Meta:
        database = db


class ExperimentResult(BaseModel):
    id_field = peewee.AutoField()

    # hyper params
    dataset_path = peewee.TextField()
    db_name_or_type = peewee.TextField()
    classifier = peewee.TextField(index=True)
    cores = peewee.IntegerField()
    output_dir = peewee.TextField()
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
    uncertainty_recommendation_certainty_threshold = peewee.FloatField(
        null=True)
    uncertainty_recommendation_ratio = peewee.FloatField(null=True)
    snuba_lite_minimum_heuristic_accuracy = peewee.FloatField(null=True)
    cluster_recommendation_minimum_cluster_unity_size = peewee.FloatField(
        null=True)
    cluster_recommendation_ratio_labeled_unlabeled = peewee.FloatField(
        null=True)
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

    param_list_id = peewee.TextField(index=True)

    cv_fit_score_mean = peewee.FloatField(null=True)
    cv_fit_score_std = peewee.FloatField(null=True)


def get_db(db_name_or_type):
    # create databases for storing the results
    if db_name_or_type == 'sqlite':
        db = peewee.SqliteDatabase('experiment_results.db')
    else:
        db = PostgresqlExtDatabase(db_name_or_type)
    db.bind([ExperimentResult])
    db.create_tables([ExperimentResult])
    #  db.connect()

    return db


def init_logging(output_dir, level=logging.INFO):
    logging_file_name = output_dir + "/" + str(
        datetime.datetime.now()) + "al_hyper_search.txt"
    if output_dir is not None:
        logging.basicConfig(
            filename=logging_file_name,
            filemode='a',
            level=level,
            format="[%(process)d] [%(asctime)s] %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=level)


def divide_data(X, Y, test_fraction):
    # split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_fraction)

    len_test = len(X_test)
    logging.info("size of test set: %i = %1.2f" %
                 (len_test, len_test / len_test))
    return X_train, X_test, Y_train, Y_test


def standard_config(additional_parameters=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--classifier',
                        default="RF",
                        help="Supported types: RF, DTree, NB, SVM, Linear")
    parser.add_argument('--cores', type=int, default=-1)
    parser.add_argument('--output_dir', default='tmp')
    parser.add_argument('--random_seed',
                        type=int,
                        default=42,
                        help="-1 Enables true Randomness")
    parser.add_argument('--test_fraction', type=float, default=0.5)

    if additional_parameters is not None:
        for additional_parameter in additional_parameters:
            parser.add_argument(*additional_parameter[0],
                                **additional_parameter[1])

    config = parser.parse_args()

    if len(sys.argv[:-1]) == 0:
        parser.print_help()
        parser.exit()

    if config.random_seed != -1:
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

    return config


def get_best_hyper_params(clf):
    if clf == "RF":
        best_hyper_params = {
            'criterion': 'gini',
            'max_depth': 46,
            'max_features': 'sqrt',
            'max_leaf_nodes': 47,
            'min_samples_leaf': 16,
            'min_samples_split': 6,
            'n_estimators': 77
        }
    elif clf == "NB":
        best_hyper_params = {'alpha': 0.7982572902331797}
    elif clf == "SVMPoly":
        best_hyper_params = {}
    elif clf == "SVMRbf":
        best_hyper_params = {
            'C': 1000,
            'cache_size': 10000,
            'gamma': 0.1,
            'kernel': 'rbf'
        }

    return best_hyper_params


def load_and_prepare_X_and_Y(dataset_path):
    # Read in dataset into pandas dataframe
    df = pd.read_csv(dataset_path, index_col="id")

    # shuffle df
    df = df.sample(frac=1).reset_index(drop=True)

    # create numpy data
    Y = df.pop('CLASS').to_numpy()

    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)

    X = df.to_numpy()

    # feature normalization
    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    # scale again to [0,1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # feature selection
    #  selector = SelectKBest(chi2, k=200)
    #  X = selector.fit_transform(X, Y)

    X = pd.DataFrame(X, dtype=float)
    Y = pd.DataFrame(Y, dtype=int)

    return X, Y, label_encoder


def train_and_evaluate(clf, X_train, Y_train, X_test, Y_test, label_encoder):
    training_times = train(clf, X_train, Y_train)
    classification_report_and_confusion_matrix(clf,
                                               X_test,
                                               Y_test,
                                               label_encoder,
                                               output_dict=False,
                                               store=True,
                                               training_times=training_times)


def train(clf, X_train, Y_train):
    f = io.StringIO()

    with contextlib.redirect_stdout(f):
        clf.fit(X_train, Y_train)

    training_times = f.getvalue()
    return training_times


def store_result(filename, content, output_dir):
    # create output folder if not existent
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + '/' + filename, 'w') as f:
        f.write(content)


def store_pickle(filename, content, output_dir):
    # create output folder if not existent
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + '/' + filename, 'wb') as f:
        pickle.dump(content, f)


def classification_report_and_confusion_matrix(clf,
                                               X,
                                               Y,
                                               label_encoder,
                                               output_dir=None,
                                               output_dict=True,
                                               store=False,
                                               training_times=""):
    Y_pred = clf.predict(X)
    clf_report = classification_report(
        Y,
        Y_pred,
        output_dict=True,
        zero_division=0,
        labels=[i for i in range(len(label_encoder.classes_))],
        target_names=label_encoder.classes_)

    conf_matrix = confusion_matrix(Y, Y_pred)

    if not output_dict:
        clf_report_string = classification_report(
            Y,
            Y_pred,
            zero_division=0,
            labels=[i for i in range(len(label_encoder.classes_))],
            target_names=label_encoder.classes_)

        logging.info(clf_report_string)
        logging.info(conf_matrix)
    else:
        return clf_report, conf_matrix

    if store:
        # create output folder if not existent
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # save Y_pred
        Y_df = pd.DataFrame(Y_pred)
        Y_df.columns = ['Y_pred']
        Y_df.insert(1, 'Y_test', Y)
        Y_df.to_csv(output_dir + '/Y_pred.csv', index=None)

        # save classification_report
        file_string = json.dumps(
            label_encoder.inverse_transform(
                [i for i in range(len(label_encoder.classes_))]).tolist())
        file_string += "\n" + "#" * 100 + "\n"
        file_string += clf_report_string
        file_string += "\n" + "#" * 100 + "\n"
        file_string += json.dumps(clf_report)
        file_string += "\n" + "#" * 100 + "\n"
        file_string += json.dumps(conf_matrix.tolist())
        file_string += "\n" + "#" * 100 + "\n"
        file_string += training_times

        store_result("results.txt", file_string, output_dir)
        store_pickle("clf.pickle", clf, output_dir)


class Logger(object):
    #source: https://stackoverflow.com/q/616645
    def __init__(self, filename="log.txt", mode="a"):
        self.stdout = sys.stdout
        self.file = open(filename, mode)
        sys.stdout = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        if self.stdout != None:
            sys.stdout = self.stdout
            self.stdout = None

        if self.file != None:
            self.file.close()
            self.file = None


def get_single_al_run_stats_table_header():
    return "Iteration: {:>3} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>3} {:>6}".format(
        "I", "L", "U", "Q", "Te", "L", "U", "SC", "SS", "QW", "CR", "QS")


def get_single_al_run_stats_row(i,
                                amount_of_labeled,
                                amount_of_unlabeled,
                                metrics_per_al_cycle,
                                index=-1):
    if amount_of_labeled == None:
        amount_of_labeled = 0
        for query_length in metrics_per_al_cycle['query_length'][:index]:
            amount_of_labeled += query_length

        amount_of_unlabeled = 2889
        for query_length in metrics_per_al_cycle['query_length'][:index]:
            amount_of_unlabeled -= query_length

    if 'accuracy' not in metrics_per_al_cycle['test_data_metrics'][0][index][
            0].keys():
        return "No test accuracy found"

    return "Iteration: {:3,d} {:6,d} {:6,d} {:6,d} {:6.1%} {:6.1%} {:6.1%} {:6.1%} {:6.1%} {:6.1%} {:>3} {:6.1%}".format(
        i,
        amount_of_labeled,
        amount_of_unlabeled,
        metrics_per_al_cycle['query_length'][index],
        metrics_per_al_cycle['test_data_metrics'][0][index][0]['accuracy'],
        metrics_per_al_cycle['train_labeled_data_metrics'][0][index][0]
        ['accuracy'],
        metrics_per_al_cycle['train_unlabeled_data_metrics'][0][index][0]
        ['accuracy'],
        metrics_per_al_cycle['stop_certainty_list'][index],
        metrics_per_al_cycle['stop_stddev_list'][index],
        metrics_per_al_cycle['stop_query_weak_accuracy_list'][index],
        metrics_per_al_cycle['recommendation'][index],
        metrics_per_al_cycle['query_strong_accuracy_list'][index],
    )


def get_toy_datasets(dataset_path):
    return None


def get_all_datasets(dataset_path):
    X = []
    Y = []
    X_data, Y_data, label_encoder = load_and_prepare_X_and_Y(dataset_path +
                                                             '/dwtc/aft.csv')
    X_train, X_test, Y_train, Y_test = divide_data(X_data,
                                                   Y_data,
                                                   test_fraction=0.5)

    X.append(
        ['dwtc', X_train, X_test, Y_train, Y_test, label_encoder.classes_])
    Y.append(None)

    for dataset_name, train_num in zip(
        ['ibn_sina', 'hiva', 'nova', 'orange', 'sylva', 'zebra'],
        [10361, 21339, 9733, 2500, 72626, 30744]):
        df = pd.read_csv(dataset_path + '/' + dataset_name + '.data',
                         header=None,
                         sep=" ")

        labels = pd.read_csv(dataset_path + '/' + dataset_name + '.label',
                             header=None)
        Y_temp = labels[0].to_numpy()
        label_encoder = LabelEncoder()
        Y_temp = label_encoder.fit_transform(Y_temp)
        X_temp = df.to_numpy()

        scaler = RobustScaler()
        X_temp = scaler.fit_transform(X_temp)

        scaler = MinMaxScaler()
        X_temp = scaler.fit_transform(X_temp)

        X_temp = pd.DataFrame(X_temp, dtype=float)
        Y_temp = pd.DataFrame(Y_temp, dtype=int)

        X_train = X_temp[:train_num]
        X_test = X_temp[train_num:]

        Y_train = Y_temp[:train_num]
        Y_test = Y_temp[train_num:]

        X.append([
            dataset_name, X_train, X_test, Y_train, Y_test,
            label_encoder.classes_
        ])
        Y.append(None)

    return X, Y
