import argparse
import contextlib
import io
import json
import os
import pickle
import random
import sys
import logging
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
import datetime
import peewee

db = peewee.SqliteDatabase('experiment_results.db')


class BaseModel(peewee.Model):
    class Meta:
        database = db


class ExperimentResult(BaseModel):
    id_field = peewee.AutoField()

    # hyper params
    dataset_path = peewee.TextField()
    classifier = peewee.TextField()
    cores = peewee.IntegerField()
    output_dir = peewee.TextField()
    test_fraction = peewee.FloatField()
    sampling = peewee.TextField()
    random_seed = peewee.IntegerField()
    cluster = peewee.TextField()
    nr_learning_iterations = peewee.IntegerField()
    nr_queries_per_iteration = peewee.IntegerField()
    start_set_size = peewee.FloatField()
    with_uncertainty_recommendation = peewee.BooleanField()
    with_cluster_recommendation = peewee.BooleanField()
    with_snuba_lite = peewee.BooleanField()
    uncertainty_recommendation_certainty_threshold = peewee.FloatField(
        null=True)
    uncertainty_recommendation_ratio = peewee.FloatField(null=True)
    snuba_lite_minimum_heuristic_accuracy = peewee.FloatField(null=True)
    cluster_recommendation_minimum_cluster_unity_size = peewee.FloatField(
        null=True)
    cluster_recommendation_ratio_labeled_unlabeled = peewee.FloatField(
        null=True)
    metrics_per_al_cycle = peewee.TextField()  # json string
    amount_of_user_asked_queries = peewee.IntegerField()
    allow_recommendations_after_stop = peewee.BooleanField()
    stopping_criteria_uncertainty = peewee.FloatField()
    stopping_criteria_acc = peewee.FloatField()
    stopping_criteria_std = peewee.FloatField()

    # information of hyperparam run
    experiment_run_date = peewee.DateTimeField(default=datetime.datetime.now)
    fit_time = peewee.TextField()  # timedelta
    confusion_matrix_test = peewee.TextField()  # json
    confusion_matrix_train = peewee.TextField()  # json
    classification_report_train = peewee.TextField()  # json
    classification_report_test = peewee.TextField()  # json
    acc_train = peewee.FloatField()
    acc_test = peewee.FloatField()
    fit_score = peewee.FloatField()


def get_db():
    # create databases for storing the results

    db.connect()
    db.create_tables([ExperimentResult])

    return db


def init_logging(output_dir):
    logging_file_name = output_dir + "/" + str(
        datetime.datetime.now()) + "al_hyper_search.txt"

    logging.basicConfig(
        filename=logging_file_name,
        filemode='a',
        level=logging.INFO,
        format="[%(process)d] [%(asctime)s] %(levelname)s: %(message)s")


def divide_data(test_fraction, start_set_size):
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
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--test_fraction', type=float, default=0.5)

    if additional_parameters is not None:
        for additional_parameter in additional_parameters:
            parser.add_argument(*additional_parameter[0],
                                **additional_parameter[1])

    config = parser.parse_args()

    if len(sys.argv[:-1]) == 0:
        parser.print_help()
        parser.exit()

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
