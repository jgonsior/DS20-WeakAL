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
import threading
from itertools import chain, combinations
from timeit import default_timer as timer

import numpy as np

#  import np.random.distributions as dists
import numpy.random
import pandas as pd
import peewee
import scipy
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from json_tricks import dumps
from playhouse.postgres_ext import *
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_covtype, load_iris
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler

db = peewee.DatabaseProxy()


class BaseModel(peewee.Model):
    class Meta:
        database = db


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
    global_score = peewee.FloatField(index=True)
    global_score_norm = peewee.FloatField(index=True)

    param_list_id = peewee.TextField(index=True)

    cv_fit_score_mean = peewee.FloatField(null=True)
    cv_fit_score_std = peewee.FloatField(null=True)

    thread_id = peewee.BigIntegerField(index=True)
    end_time = peewee.DateTimeField(index=True)


def get_db(db_name_or_type):
    # create databases for storing the results
    if db_name_or_type == "sqlite":
        db = peewee.SqliteDatabase("experiment_results.db")
    elif db_name_or_type == "tunnel":
        db = PostgresqlExtDatabase("jg",
                                   host="localhost",
                                   port=1111,
                                   password="test",
                                   user="jg")
    else:
        db = PostgresqlExtDatabase(db_name_or_type)
    db.bind([ExperimentResult])
    db.create_tables([ExperimentResult])
    #  db.connect()

    return db


#  def init_logging(output_dir, level=logging.INFO):
#  logging_file_name = output_dir + "/" + str(
#  datetime.datetime.now()) + "al_hyper_search.txt"
#  if output_dir is not None:

#  logging.basicConfig(
#  filename=logging_file_name,
#  filemode='a',
#  level=level,
#  format="[%(process)d] [%(asctime)s] %(levelname)s: %(message)s")
#  else:
#  logging.basicConfig(level=level)


def log_it(message):
    #  print(message)
    with open("tmp/log.txt", "a") as f:
        f.write("[" + str(threading.get_ident()) + "] [" +
                str(datetime.datetime.now()) + "] " + str(message) + "\n")


#  def get_logger():
#  logger = logging.getLogger('al_logger')
#  formatter = logging.Formatter(
#  "[%(process)d] [%(asctime)s] %(levelname)s: %(message)s")
#  handler = logging.FileHandler("tmp/log.txt")
#  handler.setFormatter(formatter)
#  logger.addHandler(handler)
#  return logger


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
    parser.add_argument("--datasets_path", default="../datasets/")
    parser.add_argument("--classifier",
                        default="RF",
                        help="Supported types: RF, DTree, NB, SVM, Linear")
    parser.add_argument("--cores", type=int, default=-1)
    parser.add_argument("--random_seed",
                        type=int,
                        default=42,
                        help="-1 Enables true Randomness")
    parser.add_argument("--test_fraction", type=float, default=0.5)

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
            "criterion": "gini",
            "max_depth": 46,
            "max_features": "sqrt",
            "max_leaf_nodes": 47,
            "min_samples_leaf": 16,
            "min_samples_split": 6,
            "n_estimators": 77,
        }
    elif clf == "NB":
        best_hyper_params = {"alpha": 0.7982572902331797}
    elif clf == "SVMPoly":
        best_hyper_params = {}
    elif clf == "SVMRbf":
        best_hyper_params = {
            "C": 1000,
            "cache_size": 10000,
            "gamma": 0.1,
            "kernel": "rbf",
        }

    return best_hyper_params


def load_and_prepare_X_and_Y(dataset_path):
    # Read in dataset into pandas dataframe
    df = pd.read_csv(dataset_path, index_col="id")

    # shuffle df
    df = df.sample(frac=1).reset_index(drop=True)

    # create numpy data
    Y = df.pop("CLASS").to_numpy()

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

    X = X.apply(pd.to_numeric, downcast="float", errors="ignore")
    Y = Y.apply(pd.to_numeric, downcast="integer", errors="ignore")

    return X, Y, label_encoder


def classification_report_and_confusion_matrix(clf,
                                               X,
                                               Y,
                                               label_encoder,
                                               output_dict=True,
                                               training_times=""):
    Y_pred = clf.predict(X)
    clf_report = classification_report(
        Y,
        Y_pred,
        output_dict=True,
        zero_division=0,
        labels=[i for i in range(len(label_encoder.classes_))],
        target_names=label_encoder.classes_,
    )

    conf_matrix = confusion_matrix(Y, Y_pred)

    if not output_dict:
        clf_report_string = classification_report(
            Y,
            Y_pred,
            zero_division=0,
            labels=[i for i in range(len(label_encoder.classes_))],
            target_names=label_encoder.classes_,
        )

        logging.info(clf_report_string)
        logging.info(conf_matrix)
    else:
        return clf_report, conf_matrix


class Logger(object):
    # source: https://stackoverflow.com/q/616645
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
        for query_length in metrics_per_al_cycle["query_length"][:index]:
            amount_of_labeled += query_length

        amount_of_unlabeled = 2889
        for query_length in metrics_per_al_cycle["query_length"][:index]:
            amount_of_unlabeled -= query_length

    if "accuracy" not in metrics_per_al_cycle["test_data_metrics"][0][index][
            0].keys():
        return "No test accuracy found"

    if ("accuracy" not in metrics_per_al_cycle["train_labeled_data_metrics"][0]
        [index][0].keys()):
        return "No train_labeled accuracy found"

    if ("accuracy" not in metrics_per_al_cycle["train_unlabeled_data_metrics"]
        [0][index][0].keys()):
        return "No train_unlabeled_data_metrics accuracy found"

    return "Iteration: {:3,d} {:6,d} {:6,d} {:6,d} {:6.1%} {:6.1%} {:6.1%} {:6.1%} {:6.1%} {:6.1%} {:>3} {:6.1%}".format(
        i,
        amount_of_labeled,
        amount_of_unlabeled,
        metrics_per_al_cycle["query_length"][index],
        metrics_per_al_cycle["test_data_metrics"][0][index][0]["accuracy"],
        metrics_per_al_cycle["train_labeled_data_metrics"][0][index][0]
        ["accuracy"],
        metrics_per_al_cycle["train_unlabeled_data_metrics"][0][index][0]
        ["accuracy"],
        metrics_per_al_cycle["stop_certainty_list"][index],
        metrics_per_al_cycle["stop_stddev_list"][index],
        metrics_per_al_cycle["stop_query_weak_accuracy_list"][index],
        metrics_per_al_cycle["recommendation"][index],
        metrics_per_al_cycle["query_strong_accuracy_list"][index],
    )


def prettify_bytes(bytes):
    """Get human-readable file sizes.
    simplified version of https://pypi.python.org/pypi/hurry.filesize/
    """
    # bytes pretty-printing
    units = [
        (1 << 50, " PB"),
        (1 << 40, " TB"),
        (1 << 30, " GB"),
        (1 << 20, " MB"),
        (1 << 10, " KB"),
        (1, (" byte", " bytes")),
    ]
    for factor, suffix in units:
        if bytes >= factor:
            break
    amount = int(bytes / factor)

    if isinstance(suffix, tuple):
        singular, multiple = suffix
        if amount == 1:
            suffix = singular
        else:
            suffix = multiple
    return str(amount) + suffix


def get_dataset(datasets_path, dataset_name):
    logging.info("Loading " + dataset_name)
    if dataset_name == "dwtc":
        X_data, Y_data, label_encoder = load_and_prepare_X_and_Y(
            datasets_path + "/dwtc/aft.csv")
        X_train, X_test, Y_train, Y_test = divide_data(X_data,
                                                       Y_data,
                                                       test_fraction=0.5)

        logging.info("Loaded " + dataset_name)
        return X_train, X_test, Y_train, Y_test, label_encoder.classes_

    else:
        train_indices = {
            "ibn_sina": 10361,
            "hiva": 21339,
            "nova": 9733,
            "orange": 25000,
            "sylva": 72626,
            "zebra": 30744,
        }

        if dataset_name != "forest_covtype":
            train_num = train_indices[dataset_name]

            df = pd.read_csv(
                datasets_path + "/al_challenge/" + dataset_name + ".data",
                header=None,
                sep=" ",
            )
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)

            labels = pd.read_csv(datasets_path + "/al_challenge/" +
                                 dataset_name + ".label",
                                 header=None)

            labels = labels.replace([-1], "A")
            labels = labels.replace([1], "B")
        elif dataset_name == "forest_covtype":
            X, labels = fetch_covtype(data_home=datasets_path, return_X_y=True)
            train_num = int(len(labels) / 2)
            df = pd.DataFrame(X)
            labels = pd.DataFrame(labels)

        Y_temp = labels[0].to_numpy()
        label_encoder = LabelEncoder()
        Y_temp = label_encoder.fit_transform(Y_temp)
        X_temp = df.to_numpy().astype(np.float)

        scaler = RobustScaler()
        X_temp = scaler.fit_transform(X_temp)

        scaler = MinMaxScaler()
        X_temp = scaler.fit_transform(X_temp)

        X_temp = pd.DataFrame(X_temp, dtype=float)
        Y_temp = pd.DataFrame(Y_temp, dtype=int)

        X_temp = X_temp.apply(pd.to_numeric, downcast="float", errors="ignore")
        Y_temp = Y_temp.apply(pd.to_numeric,
                              downcast="integer",
                              errors="ignore")

        X_train = X_temp[:train_num]
        X_test = X_temp[train_num:]

        Y_train = Y_temp[:train_num]
        Y_test = Y_temp[train_num:]

        logging.info("Loaded " + dataset_name)
        #  print("Size X ", prettify_bytes(sys.getsizeof(X_temp)), " \t Y ",
        #  prettify_bytes(sys.getsizeof(Y_temp)))
        return X_train, X_test, Y_train, Y_test, label_encoder.classes_


def calculate_roc_auc(label_encoder, X_test, Y_test, clf):
    #  print(set(Y_test[0].to_numpy()))
    if len(label_encoder.classes_) > 2:
        Y_scores = np.array(clf.predict_proba(X_test))
        #  print(Y_scores)
        Y_test = Y_test.to_numpy().reshape(1, len(Y_scores))[0].tolist()

        return roc_auc_score(
            Y_test,
            Y_scores,
            multi_class="ovo",
            average="macro",
            labels=[i for i in range(len(label_encoder.classes_))],
        )
    else:
        Y_scores = clf.predict_proba(X_test)[:, 1]
        #  print(Y_test.shape)
        Y_test = Y_test.to_numpy().reshape(1, len(Y_scores))[0].tolist()
        return roc_auc_score(Y_test, Y_scores)


# for details see http://www.causality.inf.ethz.ch/activelearning.php?page=evaluation#cont
def calculate_global_score(alcs, amount_of_labels_per_alcs, amount_of_labels):
    if len(alcs) > 1:
        rectangles = []
        triangles = []

        for alc, amount_of_labels_per_alc, past_alc in zip(
                alcs[1:], amount_of_labels_per_alcs[1:], alcs[:-1]):
            rectangles.append(past_alc * amount_of_labels_per_alc)
            triangles.append(
                abs(amount_of_labels_per_alc * (alc - past_alc) / 2))
        square = sum(rectangles) + sum(triangles)
    else:
        square = alcs[0] * amount_of_labels_per_alcs[0]

    amax = sum(amount_of_labels_per_alcs)
    arand = amax * (1 / amount_of_labels)
    global_score = (square - arand) / (amax - arand)

    if global_score > 1 or square < arand:
        print("alcs: ", alcs)
        print("#q: ", amount_of_labels_per_alcs)
        #  print("rect: ", rectangles)
        #  print("tria: ", triangles)
        print("ama: ", amax)
        print("ara: ", arand)
        print("squ: ", square)
        print("glob: ", global_score)

    return global_score


def get_param_distribution(hyper_search_type=None,
                           datasets_path=None,
                           classifier=None,
                           cores=None,
                           random_seed=None,
                           test_fraction=None,
                           nr_learning_iterations=None,
                           db_name_or_type=None,
                           **kwargs):
    if hyper_search_type == "random":
        zero_to_one = scipy.stats.uniform(loc=0, scale=1)
        half_to_one = scipy.stats.uniform(loc=0.5, scale=0.5)
        #  nr_queries_per_iteration = scipy.stats.randint(1, 151)
        nr_queries_per_iteration = [10]
        #  start_set_size = scipy.stats.uniform(loc=0.001, scale=0.1)
        #  start_set_size = [1, 10, 25, 50, 100]
        start_set_size = [1]
    else:
        param_size = 50
        #  param_size = 2
        zero_to_one = np.linspace(0, 1, num=param_size * 2 + 1).astype(float)
        half_to_one = np.linspace(0.5, 1, num=param_size + 1).astype(float)
        nr_queries_per_iteration = np.linspace(1, 150,
                                               num=param_size + 1).astype(int)
        #  start_set_size = np.linspace(0.001, 0.1, num=10).astype(float)
        start_set_size = [1, 10, 25, 50, 100]

    param_distribution = {
        "datasets_path": [datasets_path],
        "classifier": [classifier],
        "cores": [cores],
        "random_seed": [random_seed],
        "test_fraction": [test_fraction],
        "sampling": [
            "random",
            "uncertainty_lc",
            "uncertainty_max_margin",
            "uncertainty_entropy",
        ],
        "cluster": [
            "dummy", "random", "MostUncertain_lc", "MostUncertain_max_margin",
            "MostUncertain_entropy"
            #  'dummy',
        ],
        "nr_learning_iterations": [nr_learning_iterations],
        #  "nr_learning_iterations": [1],
        "nr_queries_per_iteration":
        nr_queries_per_iteration,
        "start_set_size":
        start_set_size,
        "stopping_criteria_uncertainty":
        zero_to_one,
        "stopping_criteria_std":
        zero_to_one,
        "stopping_criteria_acc":
        zero_to_one,
        "allow_recommendations_after_stop": [True, False],
        # uncertainty_recommendation_grid = {
        "uncertainty_recommendation_certainty_threshold":
        half_to_one,
        "uncertainty_recommendation_ratio":
        [1 / 10, 1 / 100, 1 / 1000, 1 / 10000],
        # snuba_lite_grid = {
        "snuba_lite_minimum_heuristic_accuracy": [0],
        #  half_to_one,
        # cluster_recommendation_grid = {
        "cluster_recommendation_minimum_cluster_unity_size":
        half_to_one,
        "cluster_recommendation_ratio_labeled_unlabeled":
        half_to_one,
        "with_uncertainty_recommendation": [True, False],
        "with_cluster_recommendation": [True],
        "with_snuba_lite": [False],
        "minimum_test_accuracy_before_recommendations":
        half_to_one,
        "db_name_or_type": [db_name_or_type],
    }

    return param_distribution
