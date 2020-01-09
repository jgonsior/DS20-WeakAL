import random
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.utils
from pandas.util.testing import assert_frame_equal
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight




def load_data(path):
    # reading data of files build by Data_Builder
    train_data = pd.read_csv(path + "training_data.csv", index_col=0)
    X_train_labeled = train_data.iloc[:, :-1]
    Y_train = np.array(train_data.iloc[:, -1])

    query_data = pd.read_csv(path + "query_data.csv", index_col=0)
    X_train_unlabeled = query_data.iloc[:, :-1]
    Y_query = np.array(query_data.iloc[:, -1])

    test_data = pd.read_csv(path + "test_data.csv", index_col=0)
    X_test = test_data.iloc[:, :-1]
    Y_test = np.array(test_data.iloc[:, -1])
    return (X_train_labeled, Y_train), (X_train_unlabeled, Y_query), (X_test,
                                                                      Y_test)


def calculate_accuracy_per_class(clf, classifier_classes, X_temp, Y_temp_true,
                                 list_temp):

    Y_temp_pred = clf.predict(X_temp)

    accuracies = []

    for classifier_class in classifier_classes:
        #retrieve indices of
        # a = [i for i, s in enumerate(Y_temp_true) if classifier_class in s]
        a = np.where(Y_temp_true == classifier_class)[0]
        if len(a) != 0:
            class_accuracy = np.count_nonzero(
                Y_temp_pred[a] == classifier_class) / len(a)
        else:
            class_accuracy = 0
        accuracies.append(class_accuracy)
        #print ("\t accuracy for %s is: %r " % (classifier_class, class_accuracy))
    return accuracies


def print_clf_comparison(
    clf_active,
    clf_passive_starter,
    clf_passive_full,
    len_active,
    features_path,
    meta_path,
    classifier_classes,
    target_names,
    start_size,
    X_train_labeled_orig,
    X_train_unlabeled_orig,
    X_test_orig,
    Y_train_orig,
    Y_query_orig,
    Y_test_orig,
    merged_labels=False,
    random_state=None,
):

    np.random.seed(random_state)
    random.seed(random_state)
    # instantiate data

    ((X_train_labeled, Y_train), (X_train_unlabeled, Y_query),
     (X_test, Y_test), _), _ = load_query_data(features_path,
                                               meta_path,
                                               start_size,
                                               merged_labels,
                                               random_state=random_state)

    if not X_train_labeled.equals(X_train_labeled_orig):
        print("X_train_labeled")

    if not X_train_unlabeled.equals(X_train_unlabeled_orig):
        print("X_train_unlabeled")

    if not X_test.equals(X_test_orig):
        print("X_test")

    if not np.size(Y_train) - np.count_nonzero(np.equal(Y_train,
                                                        Y_train_orig)) == 0:
        print("Y_train")

    if not np.size(Y_query) - np.count_nonzero(np.equal(Y_query,
                                                        Y_query_orig)) == 0:
        print("Y_query")

    if not np.size(Y_test) - np.count_nonzero(np.equal(Y_test,
                                                       Y_test_orig)) == 0:
        print("Y_test")

    X_train_labeled = X_train_labeled_orig
    X_train_unlabeled = X_train_unlabeled_orig
    X_test = X_test_orig
    Y_train = Y_train_orig
    Y_query = Y_query_orig
    Y_test = Y_test_orig

    # building data for full training set
    Y_train_full = np.append(Y_train, Y_query)
    X_train_labeled_full = X_train_labeled.append(X_train_unlabeled)

    # first combine X_train_labeled_full and Y_train_full
    # then sort it after index
    # then shuffle it so that the ordering is the same for all iterations
    X_train_labeled_full['labels'] = Y_train_full
    X_train_labeled_full.sort_index(inplace=True)
    X_train_labeled_full = sklearn.utils.shuffle(X_train_labeled_full,
                                                 random_state=random_state)
    Y_train_full = X_train_labeled_full['labels']
    del X_train_labeled_full['labels']

    print("Länge X_train_labeled:" + str(len(X_train_labeled)))
    print("Länge X_train_unlabeled:" + str(len(X_train_unlabeled)))
    print("Länge X_full:" + str(len(X_train_labeled_full)))

    print("Länge Y_train:" + str(len(Y_train)))
    print("Länge Y_query:" + str(len(Y_query)))
    print("Länge Y_full:" + str(len(Y_train_full)))
    pprint((X_train_labeled_full.index))
    # calculate sizes
    len_full = len(X_train_unlabeled) + len(X_train_labeled)
    len_starter = len(X_train_labeled)

    # testing classifier learning starter training set
    print("number of samples: %i" % len_starter)

    clf_passive_starter.fit(X_train_labeled,
                            Y_train,
                            sample_weight=compute_sample_weight(
                                'balanced', Y_train))
    Y_pred_starter = clf_passive_starter.predict(X_test)
    accuracy_starter = accuracy_score(Y_test, Y_pred_starter)
    print(
        "accuracy of simple classifier training \033[1m starter set\033[0m  : %1.3f"
        % accuracy_starter)
    print("using %1.2f %% of possible training data \n" %
          (len_starter / len_full * 100))

    # testing classifier learning full training set
    print("number of samples: %i" % len_full)

    clf_passive_full.fit(X_train_labeled_full,
                         Y_train_full,
                         sample_weight=compute_sample_weight(
                             'balanced', Y_train_full))
    Y_pred_full = clf_passive_full.predict(X_test)
    accuracy_full = accuracy_score(Y_test, Y_pred_full)
    print(
        "accuracy of simple classifier training \033[1m biggest possible training set \033[0m: %1.3f"
        % accuracy_full)
    print("using 100 % of possible training data \n")

    # active learning classifier
    print("number of samples: %i" % (len_active))

    Y_pred_active = clf_active.predict(X_test)
    accuracy_active = accuracy_score(Y_test, Y_pred_active)
    print("accuracy of \033[1m active classifier \033[0m: %1.3f" %
          accuracy_active)
    print("using %1.2f %%  of possible training data \n" %
          (len_active / len_full * 100))

    print("\n")

    #for full trained classifier
    print(
        "confusion Matrix of simple classifier using biggest possible training set"
    )

    classificationReport = classification_report(Y_test,
                                                 Y_pred_full,
                                                 target_names=target_names)
    print(classificationReport)
    print(confusion_matrix(Y_test, Y_pred_full))

    #for starter classifier
    print(
        "\n confusion Matrix of simple classifier using starter training set")

    classificationReport = classification_report(Y_test,
                                                 Y_pred_starter,
                                                 target_names=target_names)
    print(classificationReport)
    print(confusion_matrix(Y_test, Y_pred_starter))

    # for active classifier
    print(
        "\n confusion Matrix of active classifier using starter training set and query iterations"
    )

    classificationReport = classification_report(Y_test,
                                                 Y_pred_active,
                                                 target_names=target_names)
    print(classificationReport)
    print(confusion_matrix(Y_test, Y_pred_active))


def plot_results(accuracy_list, class_accuracy_list, classifier_classes,
                 certainty_list, nr_learning_iterations,
                 nr_queries_per_iteration):
    # plotting accuracy
    plt.ion()
    x = np.array(range(0, nr_learning_iterations)) * nr_queries_per_iteration
    A = np.ma.array(accuracy_list)
    plt.ylabel('accuracy in %')
    plt.xlabel('queries')
    plt.plot(x, A.T)
    plt.legend(("test set", "training set"), loc=0)
    plt.show()

    # plot all accuracies per class for each itteration
    plt.plot(np.array(range(nr_learning_iterations)) *
             nr_queries_per_iteration,
             class_accuracy_list,
             label=classifier_classes)
    plt.ylabel('accuracy in %')
    plt.xlabel('queries')
    for i in range(len(classifier_classes)):
        plt.legend(classifier_classes, loc=0)
    plt.show()

    # plotting accuracy
    plt.ion()
    x = np.array(range(0, nr_learning_iterations)) * nr_queries_per_iteration
    A = np.ma.array(certainty_list)
    plt.ylabel('certainty  in %')
    plt.xlabel('queries')
    plt.plot(x, A.T)
    plt.legend(("min", "mean", "max"), loc=0)
    plt.show()
