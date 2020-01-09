import argparse
import contextlib
import io
import json
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler


def standard_config(additional_parameters=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--classifier',
                        default="RF",
                        help="Supported types: RF, DTree, NB, SVM, Linear")
    parser.add_argument('--cores', type=int, default=-1)
    parser.add_argument('--output_dir', default='tmp/')
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


def load_and_prepare_X_and_Y(config):
    # Read in dataset into pandas dataframe
    df = pd.read_csv(config.dataset_path, index_col="id")

    # shuffle df
    df = df.sample(frac=1,
                   random_state=config.random_seed).reset_index(drop=True)

    # create numpy data
    Y = df.pop('CLASS').to_numpy()

    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)

    X = df.to_numpy()

    # feature normalization
    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    # scale again to [0,1]
    #  scaler = MinMaxScaler()
    #  X = scaler.fit_transform(X)

    # feature selection
    #  selector = SelectKBest(chi2, k=200)
    #  X = selector.fit_transform(X, Y)

    return X, Y, label_encoder


def train_and_evaluate(clf, X_train, Y_train, X_test, Y_test, config,
                       label_encoder):
    training_times = train(clf, X_train, Y_train)
    classification_report_and_confusion_matrix(clf,
                                               X_test,
                                               Y_test,
                                               config,
                                               label_encoder,
                                               store=True,
                                               training_times=training_times)


def train(clf, X_train, Y_train):
    f = io.StringIO()

    with contextlib.redirect_stdout(f):
        clf.fit(X_train, Y_train)

    training_times = f.getvalue()
    return training_times


def classification_report_and_confusion_matrix(clf,
                                               X_test,
                                               Y_test,
                                               config,
                                               label_encoder,
                                               output_dict=True,
                                               store=False,
                                               training_times=""):

    Y_pred = clf.predict(X_test)
    clf_report = classification_report(Y_test,
                                       Y_pred,

                                       output_dict=True,
                                       target_names=label_encoder.classes_)

    conf_matrix = confusion_matrix(Y_test, Y_pred)

    if not output_dict:
        clf_report_string = classification_report(
            Y_test, Y_pred, target_names=label_encoder.classes_)

        print(clf_report_string)
        print(conf_matrix)
    else:
        return clf_report, conf_matrix

    if store:

        # create output folder if not existent
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)

        # save Y_pred
        Y_df = pd.DataFrame(Y_pred)
        Y_df.columns = ['Y_pred']
        Y_df.insert(1, 'Y_test', Y_test)
        Y_df.to_csv(config.output_dir + '/Y_pred.csv', index=None)

        # save classification_report
        with open(config.output_dir + '/results.txt', 'w') as f:
            f.write(
                json.dumps(
                    label_encoder.inverse_transform([
                        i for i in range(len(label_encoder.classes_))
                    ]).tolist()))
            f.write("\n" + "#" * 100 + "\n")
            f.write(clf_report_string)
            f.write("\n" + "#" * 100 + "\n")
            f.write(json.dumps(clf_report))
            f.write("\n" + "#" * 100 + "\n")
            f.write(json.dumps(conf_matrix.tolist()))
            f.write("\n" + "#" * 100 + "\n")
            f.write(training_times)

        with open(config.output_dir + '/clf.pickle', 'wb') as f:
            pickle.dump(clf, f)



def print_data_segmentation(X_train_labeled, X_train_unlabeled, X_test,
                            len_queries):
    len_train_labeled = len(X_train_labeled)
    len_train_unlabeled = len(X_train_unlabeled)
    len_test = len(X_test)

    len_total = len_train_unlabeled + len_train_labeled + len_trains

    print("size of train set: %i = %1.2f" %
          (len_train_labeled, len_train_labeled / len_total))
    print("size of query set: %i = %1.2f" %
          (len_train_unlabeled, len_train_unlabeled / len_total))
    print("learned %i queries of query set" % (len_train_unlabeled))
    print("size of test set: %i = %1.2f" %
          (len_testset, len_testset / len_total))
