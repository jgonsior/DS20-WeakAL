from sklearn.metrics import accuracy_score
import argparse
import datetime
import os
import random
import sys
import threading

import numpy as np

#  import np.random.distributions as dists
import numpy.random
import pandas as pd
import scipy
from sklearn.datasets import fetch_covtype, make_classification
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler

# really dirty hack to provide logging as functions instead of objects
def init_logger(logfilepath):
    global logfile_path
    logfile_path = logfilepath


def log_it(message):
    message = (
        "["
        + str(threading.get_ident())
        + "] ["
        + str(datetime.datetime.now())
        + "] "
        + str(message)
    )

    if logfile_path == "console":
        print(message)
    else:
        with open(logfile_path, "a") as f:
            f.write(message + "\n")


def standard_config(additional_parameters=None, standard_args=True):
    parser = argparse.ArgumentParser()
    if standard_args:
        parser.add_argument("--DATASETS_PATH", default="../datasets/")
        parser.add_argument(
            "--CLASSIFIER",
            default="RF",
            help="Supported types: RF, DTree, NB, SVM, Linear",
        )
        parser.add_argument("--N_JOBS", type=int, default=-1)
        parser.add_argument(
            "--RANDOM_SEED", type=int, default=42, help="-1 Enables true Randomness"
        )
        parser.add_argument("--TEST_FRACTION", type=float, default=0.5)
        parser.add_argument("--LOG_FILE", type=str, default="log.txt")

    if additional_parameters is not None:
        for additional_parameter in additional_parameters:
            parser.add_argument(*additional_parameter[0], **additional_parameter[1])

    config = parser.parse_args()

    if len(sys.argv[:-1]) == 0:
        parser.print_help()
        parser.exit()

    if config.RANDOM_SEED != -1 and config.RANDOM_SEED != -2:
        np.random.seed(config.RANDOM_SEED)
        random.seed(config.RANDOM_SEED)

    init_logger(config.LOG_FILE)

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


def conf_matrix_and_acc(clf, X, Y_true, label_encoder):
    Y_pred = clf.predict(X)
    conf_matrix = confusion_matrix(Y_true, Y_pred)
    acc = accuracy_score(Y_true, Y_pred)
    return conf_matrix, acc


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
    return "Iteration: {:>3} {:>6} {:>6} {:>6} {:>6} {:>6} {:>3}".format(
        "I", "L", "U", "Q", "Te", "Tr", "R"
    )


def get_single_al_run_stats_row(
    i, amount_of_labeled, amount_of_unlabeled, metrics_per_al_cycle, index=-1
):
    if amount_of_labeled == None:
        amount_of_labeled = 0
        for query_length in metrics_per_al_cycle["query_length"][:index]:
            amount_of_labeled += query_length

        amount_of_unlabeled = 2889
        for query_length in metrics_per_al_cycle["query_length"][:index]:
            amount_of_unlabeled -= query_length

    return "Iteration: {:3,d} {:6,d} {:6,d} {:6,d} {:6.1%} {:6.1%} {:>3}".format(
        i,
        amount_of_labeled,
        amount_of_unlabeled,
        metrics_per_al_cycle["query_length"][index],
        metrics_per_al_cycle["test_acc"][index],
        metrics_per_al_cycle["train_acc"][index],
        metrics_per_al_cycle["recommendation"][index],
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


def get_dataset(datasets_path, dataset_name, RANDOM_SEED, **kwargs):
    log_it("Loading " + dataset_name)

    if dataset_name == "dwtc":
        df = pd.read_csv(datasets_path + "/dwtc/aft.csv", index_col="id")

        # shuffle df
        df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        Y_temp = df.pop("CLASS").to_numpy()

        # replace labels with strings
        #  Y_temp = Y_temp.astype("str")
        #  for i in range(0, 40):
        #  np.place(Y_temp, Y_temp == str(i), chr(65 + i))
    elif dataset_name == "synthetic":
        X_data, Y_temp = make_classification(**kwargs)
        df = pd.DataFrame(X_data)

        # replace labels with strings
        Y_temp = Y_temp.astype("str")
        for i in range(0, kwargs["n_classes"]):
            np.place(Y_temp, Y_temp == str(i), chr(65 + i))

    elif dataset_name == "forest_covtype":
        X, Y_temp = fetch_covtype(data_home=datasets_path, return_X_y=True)
        train_num = int(len(labels) / 2)
        df = pd.DataFrame(X)
    else:
        df = pd.read_csv(
            datasets_path + "/al_challenge/" + dataset_name + ".data",
            header=None,
            sep=" ",
        )

        # shuffle df
        df = df.sample(frac=1, random_state=RANDOM_SEED)

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        labels = pd.read_csv(
            datasets_path + "/al_challenge/" + dataset_name + ".label", header=None
        )

        labels = labels.replace([-1], "A")
        labels = labels.replace([1], "B")
        Y_temp = labels[0].to_numpy()

    train_indices = {
        "ibn_sina": 10361,
        "hiva": 21339,
        "nova": 9733,
        "orange": 25000,
        "sylva": 72626,
        "zebra": 30744,
    }

    if dataset_name in train_indices:
        train_num = train_indices[dataset_name]
    else:
        train_num = int(len(Y_temp) * 0.5)

    label_encoder = LabelEncoder()
    Y_temp = label_encoder.fit_transform(Y_temp)
    X_temp = df.to_numpy().astype(np.float)

    # feature normalization
    scaler = RobustScaler()
    X_temp = scaler.fit_transform(X_temp)

    # scale back to [0,1]
    scaler = MinMaxScaler()
    X_temp = scaler.fit_transform(X_temp)

    X_temp = pd.DataFrame(X_temp, dtype=float)
    Y_temp = pd.DataFrame(Y_temp, dtype=int)

    X_temp = X_temp.apply(pd.to_numeric, downcast="float", errors="ignore")
    Y_temp = Y_temp.apply(pd.to_numeric, downcast="integer", errors="ignore")

    X_train = X_temp[:train_num]
    X_test = X_temp[train_num:]

    Y_train = Y_temp[:train_num]
    Y_test = Y_temp[train_num:]

    log_it("Loaded " + dataset_name)
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
def calculate_global_score(
    metric_values, amount_of_labels_per_metric_values, amount_of_labels
):
    if len(metric_values) > 1:
        rectangles = []
        triangles = []

        for (
            metric_value,
            amount_of_labels_per_metric_value,
            past_metric_value,
        ) in zip(
            metric_values[1:],
            amount_of_labels_per_metric_values[1:],
            metric_values[:-1],
        ):
            rectangles.append(metric_value * amount_of_labels_per_metric_value)
            triangles.append(
                amount_of_labels_per_metric_value
                * (past_metric_value - metric_value)
                / 2
            )
        square = sum(rectangles) + sum(triangles)
    else:
        square = metric_values[0] * amount_of_labels_per_metric_values[0]

    amax = sum(amount_of_labels_per_metric_values)
    arand = amax * (1 / amount_of_labels)
    global_score = (square - arand) / (amax - arand)

    if global_score > 1:
        print("metric_values: ", metric_values)
        print("#q: ", amount_of_labels_per_metric_values)
        print("rect: ", rectangles)
        print("tria: ", triangles)
        print("ama: ", amax)
        print("ara: ", arand)
        print("squ: ", square)
        print("glob: ", global_score)

    return global_score


def get_param_distribution(
    hyper_search_type=None,
    DATASETS_PATH=None,
    CLASSIFIER=None,
    N_JOBS=None,
    RANDOM_SEED=None,
    TEST_FRACTION=None,
    NR_LEARNING_ITERATIONS=None,
    OUTPUT_DIRECTORY=None,
    **kwargs
):
    if hyper_search_type == "random":
        zero_to_one = scipy.stats.uniform(loc=0, scale=1)
        half_to_one = scipy.stats.uniform(loc=0.5, scale=0.5)
        #  nr_queries_per_iteration = scipy.stats.randint(1, 151)
        NR_QUERIES_PER_ITERATION = [10]
        #  START_SET_SIZE = scipy.stats.uniform(loc=0.001, scale=0.1)
        #  START_SET_SIZE = [1, 10, 25, 50, 100]
        START_SET_SIZE = [1]
    else:
        param_size = 50
        #  param_size = 2
        zero_to_one = np.linspace(0, 1, num=param_size * 2 + 1).astype(float)
        half_to_one = np.linspace(0.5, 1, num=param_size + 1).astype(float)
        NR_QUERIES_PER_ITERATION = [
            10
        ]  # np.linspace(1, 150, num=param_size + 1).astype(int)
        #  START_SET_SIZE = np.linspace(0.001, 0.1, num=10).astype(float)
        START_SET_SIZE = [1]

    param_distribution = {
        "DATASETS_PATH": [DATASETS_PATH],
        "CLASSIFIER": [CLASSIFIER],
        "N_JOBS": [N_JOBS],
        "RANDOM_SEED": [RANDOM_SEED],
        "TEST_FRACTION": [TEST_FRACTION],
        "SAMPLING": [
            "random",
            "uncertainty_lc",
            "uncertainty_max_margin",
            "uncertainty_entropy",
        ],
        "CLUSTER": [
            "dummy",
            "random",
            "MostUncertain_lc",
            "MostUncertain_max_margin",
            "MostUncertain_entropy"
            #  'dummy',
        ],
        "NR_LEARNING_ITERATIONS": [NR_LEARNING_ITERATIONS],
        #  "NR_LEARNING_ITERATIONS": [1],
        "NR_QUERIES_PER_ITERATION": NR_QUERIES_PER_ITERATION,
        "START_SET_SIZE": START_SET_SIZE,
        "STOPPING_CRITERIA_UNCERTAINTY": [1],  # zero_to_one,
        "STOPPING_CRITERIA_STD": [1],  # zero_to_one,
        "STOPPING_CRITERIA_ACC": [1],  # zero_to_one,
        "ALLOW_RECOMMENDATIONS_AFTER_STOP": [True],
        # uncertainty_recommendation_grid = {
        "UNCERTAINTY_RECOMMENDATION_CERTAINTY_THRESHOLD": np.linspace(
            0.85, 1, num=15 + 1
        ),  # half_to_one,
        "UNCERTAINTY_RECOMMENDATION_RATIO": [
            1 / 100,
            1 / 1000,
            1 / 10000,
            1 / 100000,
            1 / 1000000,
        ],
        # snuba_lite_grid = {
        "SNUBA_LITE_MINIMUM_HEURISTIC_ACCURACY": [0],
        #  half_to_one,
        # cluster_recommendation_grid = {
        "CLUSTER_RECOMMENDATION_MINIMUM_CLUSTER_UNITY_SIZE": half_to_one,
        "CLUSTER_RECOMMENDATION_RATIO_LABELED_UNLABELED": half_to_one,
        "WITH_UNCERTAINTY_RECOMMENDATION": [True, False],
        "WITH_CLUSTER_RECOMMENDATION": [True, False],
        "WITH_SNUBA_LITE": [False],
        "MINIMUM_TEST_ACCURACY_BEFORE_RECOMMENDATIONS": half_to_one,
        "OUTPUT_DIRECTORY": [OUTPUT_DIRECTORY],
        "USER_QUERY_BUDGET_LIMIT": [200],
    }

    return param_distribution
