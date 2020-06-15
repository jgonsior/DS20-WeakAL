import datetime
import hashlib
import math
import operator
import threading
from timeit import default_timer as timer

#  import np.random.distributions as dists
from json_tricks import dumps
from sklearn.preprocessing import LabelEncoder

from .cluster_strategies import (
    DummyClusterStrategy,
    MostUncertainClusterStrategy,
    RandomClusterStrategy,
    RoundRobinClusterStrategy,
)
from .dataStorage import DataStorage
from .experiment_setup_lib import (
    ExperimentResult,
    calculate_global_score,
    classification_report_and_confusion_matrix,
    get_db,
    get_param_distribution,
    init_logger,
)
from .sampling_strategies import BoundaryPairSampler, RandomSampler, UncertaintySampler


def train_al(
    X_labeled,
    Y_labeled,
    X_unlabeled,
    label_encoder,
    START_SET_SIZE,
    hyper_parameters,
    oracle,
    TEST_FRACTION=None,
    X_test=None,
    Y_test=None,
):
    hyper_parameters["LEN_TRAIN_DATA"] = len(X_labeled)
    dataset_storage = DataStorage(hyper_parameters["RANDOM_SEED"])
    dataset_storage.set_training_data(
        X_labeled,
        Y_labeled,
        X_unlabeled=X_unlabeled,
        START_SET_SIZE=START_SET_SIZE,
        TEST_FRACTION=TEST_FRACTION,
        label_encoder=label_encoder,
        hyper_parameters=hyper_parameters,
        X_test=X_test,
        Y_test=Y_test,
    )

    if hyper_parameters["CLUSTER"] == "dummy":
        cluster_strategy = DummyClusterStrategy()
    elif hyper_parameters["CLUSTER"] == "random":
        cluster_strategy = RandomClusterStrategy()
    elif hyper_parameters["CLUSTER"] == "MostUncertain_lc":
        cluster_strategy = MostUncertainClusterStrategy()
        cluster_strategy.set_uncertainty_strategy("least_confident")
    elif hyper_parameters["CLUSTER"] == "MostUncertain_max_margin":
        cluster_strategy = MostUncertainClusterStrategy()
        cluster_strategy.set_uncertainty_strategy("max_margin")
    elif hyper_parameters["CLUSTER"] == "MostUncertain_entropy":
        cluster_strategy = MostUncertainClusterStrategy()
        cluster_strategy.set_uncertainty_strategy("entropy")
    elif hyper_parameters["CLUSTER"] == "RoundRobin":
        cluster_strategy = RoundRobinClusterStrategy()

    cluster_strategy.set_data_storage(dataset_storage, hyper_parameters["N_JOBS"])

    active_learner_params = {
        "dataset_storage": dataset_storage,
        "cluster_strategy": cluster_strategy,
        "N_JOBS": hyper_parameters["N_JOBS"],
        "RANDOM_SEED": hyper_parameters["RANDOM_SEED"],
        "NR_LEARNING_ITERATIONS": hyper_parameters["NR_LEARNING_ITERATIONS"],
        "NR_QUERIES_PER_ITERATION": hyper_parameters["NR_QUERIES_PER_ITERATION"],
        "oracle": oracle,
    }

    if hyper_parameters["SAMPLING"] == "random":
        active_learner = RandomSampler(**active_learner_params)
    elif hyper_parameters["SAMPLING"] == "boundary":
        active_learner = BoundaryPairSampler(**active_learner_params)
    elif hyper_parameters["SAMPLING"] == "uncertainty_lc":
        active_learner = UncertaintySampler(**active_learner_params)
        active_learner.set_uncertainty_strategy("least_confident")
    elif hyper_parameters["SAMPLING"] == "uncertainty_max_margin":
        active_learner = UncertaintySampler(**active_learner_params)
        active_learner.set_uncertainty_strategy("max_margin")
    elif hyper_parameters["SAMPLING"] == "uncertainty_entropy":
        active_learner = UncertaintySampler(**active_learner_params)
        active_learner.set_uncertainty_strategy("entropy")
    #  elif hyper_parameters['sampling'] == 'committee':
    #  active_learner = CommitteeSampler(hyper_parameters['RANDOM_SEED, hyper_parameters.N_JOBS, hyper_parameters.NR_LEARNING_ITERATIONS)
    else:
        ("No Active Learning Strategy specified")

    start = timer()
    trained_active_clf_list, metrics_per_al_cycle, Y_train = active_learner.learn(
        **hyper_parameters
    )
    end = timer()

    return (
        trained_active_clf_list,
        Y_train,
        end - start,
        metrics_per_al_cycle,
        dataset_storage,
        active_learner,
    )


def eval_al(
    X_test,
    Y_test,
    label_encoder,
    trained_active_clf_list,
    fit_time,
    metrics_per_al_cycle,
    dataset_storage,
    active_learner,
    hyper_parameters,
    dataset_name,
):
    hyper_parameters[
        "amount_of_user_asked_queries"
    ] = active_learner.amount_of_user_asked_queries

    classification_report_and_confusion_matrix_test = classification_report_and_confusion_matrix(
        trained_active_clf_list[0], X_test, Y_test, dataset_storage.label_encoder
    )
    classification_report_and_confusion_matrix_train = classification_report_and_confusion_matrix(
        trained_active_clf_list[0],
        dataset_storage.X_train_labeled,
        dataset_storage.Y_train_labeled,
        dataset_storage.label_encoder,
    )

    # normalize by start_set_size
    percentage_user_asked_queries = (
        1
        - hyper_parameters["amount_of_user_asked_queries"]
        / hyper_parameters["LEN_TRAIN_DATA"]
    )
    test_acc = classification_report_and_confusion_matrix_test[0]["accuracy"]

    # score is harmonic mean
    score = (
        2
        * percentage_user_asked_queries
        * test_acc
        / (percentage_user_asked_queries + test_acc)
    )

    amount_of_labels = len(label_encoder.classes_)

    acc_with_weak_values = [
        metrics_per_al_cycle["test_data_metrics"][0][i][0]["accuracy"]
        for i in range(0, len(metrics_per_al_cycle["query_length"]))
    ]
    roc_auc_with_weak_values = metrics_per_al_cycle["all_unlabeled_roc_auc_scores"]
    acc_with_weak_amount_of_labels = (
        roc_auc_with_weak_amount_of_labels
    ) = metrics_per_al_cycle["query_length"]
    acc_with_weak_amount_of_labels_norm = roc_auc_with_weak_amount_of_labels_norm = [
        math.log2(m) for m in acc_with_weak_amount_of_labels
    ]

    # no recommendation indices
    no_weak_indices = [
        i
        for i, j in enumerate(metrics_per_al_cycle["recommendation"])
        if j == "A" or j == "G"
    ]

    if no_weak_indices == [0]:
        no_weak_indices.append(0)

    acc_no_weak_values = operator.itemgetter(*no_weak_indices)(acc_with_weak_values)
    roc_auc_no_weak_values = operator.itemgetter(*no_weak_indices)(
        roc_auc_with_weak_values
    )
    acc_no_weak_amount_of_labels = (
        roc_auc_no_weak_amount_of_labels
    ) = operator.itemgetter(*no_weak_indices)(acc_with_weak_amount_of_labels)
    acc_no_weak_amount_of_labels_norm = (
        roc_auc_no_weak_amount_of_labels_norm
    ) = operator.itemgetter(*no_weak_indices)(acc_with_weak_amount_of_labels_norm)

    global_score_no_weak_roc_auc = calculate_global_score(
        roc_auc_no_weak_values, roc_auc_no_weak_amount_of_labels, amount_of_labels
    )
    global_score_no_weak_roc_auc_norm = calculate_global_score(
        roc_auc_no_weak_values, roc_auc_no_weak_amount_of_labels_norm, amount_of_labels
    )
    global_score_no_weak_acc = calculate_global_score(
        acc_no_weak_values, acc_no_weak_amount_of_labels, amount_of_labels
    )
    global_score_no_weak_acc_norm = calculate_global_score(
        acc_no_weak_values, acc_no_weak_amount_of_labels_norm, amount_of_labels
    )

    global_score_with_weak_roc_auc = calculate_global_score(
        roc_auc_with_weak_values, roc_auc_with_weak_amount_of_labels, amount_of_labels
    )
    global_score_with_weak_roc_auc_norm = calculate_global_score(
        roc_auc_with_weak_values,
        roc_auc_with_weak_amount_of_labels_norm,
        amount_of_labels,
    )
    global_score_with_weak_acc = calculate_global_score(
        acc_with_weak_values, acc_with_weak_amount_of_labels, amount_of_labels
    )
    global_score_with_weak_acc_norm = calculate_global_score(
        acc_with_weak_values, acc_with_weak_amount_of_labels_norm, amount_of_labels
    )

    # calculate based on params a unique id which should be the same across all similar cross validation splits
    param_distribution = get_param_distribution(**hyper_parameters)
    unique_params = ""
    for k in param_distribution.keys():
        unique_params += str(hyper_parameters[k])

    param_list_id = hashlib.md5(unique_params.encode("utf-8")).hexdigest()
    db = get_db(db_name_or_type=hyper_parameters["DB_NAME_OR_TYPE"])

    hyper_parameters["DATASET_NAME"] = dataset_name
    print(hyper_parameters.keys())
    hyper_parameters["cores"] = hyper_parameters["N_JOBS"]
    #  del hyper_parameters["N_JOBS"]

    # lower case all parameters for nice values in database
    hyper_parameters = {k.lower(): v for k, v in hyper_parameters.items()}

    experiment_result = ExperimentResult(
        **hyper_parameters,
        metrics_per_al_cycle=dumps(metrics_per_al_cycle, allow_nan=True),
        fit_time=str(fit_time),
        confusion_matrix_test=dumps(
            classification_report_and_confusion_matrix_test[1], allow_nan=True
        ),
        confusion_matrix_train=dumps(
            classification_report_and_confusion_matrix_train[1], allow_nan=True
        ),
        classification_report_test=dumps(
            classification_report_and_confusion_matrix_test[0], allow_nan=True
        ),
        classification_report_train=dumps(
            classification_report_and_confusion_matrix_train[0], allow_nan=True
        ),
        acc_train=classification_report_and_confusion_matrix_train[0]["accuracy"],
        acc_test=classification_report_and_confusion_matrix_test[0]["accuracy"],
        fit_score=score,
        roc_auc=metrics_per_al_cycle["all_unlabeled_roc_auc_scores"][-1],
        param_list_id=param_list_id,
        thread_id=threading.get_ident(),
        end_time=datetime.datetime.now(),
        global_score_no_weak_roc_auc=global_score_no_weak_roc_auc,
        global_score_no_weak_roc_auc_norm=global_score_no_weak_roc_auc_norm,
        global_score_no_weak_acc=global_score_no_weak_acc,
        global_score_no_weak_acc_norm=global_score_no_weak_acc_norm,
        global_score_with_weak_roc_auc=global_score_with_weak_roc_auc,
        global_score_with_weak_roc_auc_norm=global_score_with_weak_roc_auc_norm,
        global_score_with_weak_acc=global_score_with_weak_acc,
        global_score_with_weak_acc_norm=global_score_with_weak_acc_norm,
        global_score_with_weak_roc_auc_old=0,
        global_score_with_weak_roc_auc_norm_old=0,
    )
    experiment_result.save()
    db.close()
    return score


"""
Takes a dataset_path, X, Y, label_encoder and does the following steps:
1. Split data
2. Train AL on the train dataset
3. Evaluate AL on the test dataset
4. Returns fit_score
"""


def train_and_eval_dataset(
    dataset_name,
    X_train,
    X_test,
    Y_train,
    Y_test,
    label_encoder_classes,
    hyper_parameters,
    oracle,
):
    init_logger("log.txt")
    label_encoder = LabelEncoder()
    label_encoder.fit(label_encoder_classes)

    (
        trained_active_clf_list,
        Y_train,
        fit_time,
        metrics_per_al_cycle,
        dataStorage,
        active_learner,
    ) = train_al(
        X_train,
        Y_train,
        X_unlabeled=None,
        label_encoder=label_encoder,
        START_SET_SIZE=hyper_parameters["START_SET_SIZE"],
        hyper_parameters=hyper_parameters,
        oracle=oracle,
        TEST_FRACTION=hyper_parameters["TEST_FRACTION"],
        X_test=X_test,
        Y_test=Y_test,
    )

    fit_score = eval_al(
        X_test,
        Y_test,
        label_encoder,
        trained_active_clf_list,
        fit_time,
        metrics_per_al_cycle,
        dataStorage,
        active_learner,
        hyper_parameters,
        dataset_name,
    )
    return fit_score
