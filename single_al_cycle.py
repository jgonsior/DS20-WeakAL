from al_cycle_wrapper import train_and_eval_dataset
from experiment_setup_lib import (
    get_dataset,
    standard_config,
)

config = standard_config(
    [
        (
            ["--SAMPLING"],
            {
                "required": True,
                "help": "Possible values: uncertainty, random, committe, boundary",
            },
        ),
        (["--DATASET_NAME"], {"required": True,}),
        (
            ["--CLUSTER"],
            {
                "default": "dummy",
                "help": "Possible values: dummy, random, mostUncertain, roundRobin",
            },
        ),
        (["--NR_LEARNING_ITERATIONS"], {"type": int, "default": 150000}),
        (["--NR_QUERIES_PER_ITERATION"], {"type": int, "default": 150}),
        (["--START_SET_SIZE"], {"type": int, "default": 1}),
        (
            ["--MINIMUM_TEST_ACCURACY_BEFORE_RECOMMENDATIONS"],
            {"type": float, "default": 0.7},
        ),
        (
            ["--UNCERTAINTY_RECOMMENDATION_CERTAINTY_THRESHOLD"],
            {"type": float, "default": 0.9},
        ),
        (["--UNCERTAINTY_RECOMMENDATION_RATIO"], {"type": float, "default": 1 / 100}),
        (["--SNUBA_LITE_MINIMUM_HEURISTIC_ACCURACY"], {"type": float, "default": 0.9}),
        (
            ["--CLUSTER_RECOMMENDATION_MINIMUM_CLUSTER_UNITY_SIZE"],
            {"type": float, "default": 0.7},
        ),
        (
            ["--CLUSTER_RECOMMENDATION_RATIO_LABELED_UNLABELED"],
            {"type": float, "default": 0.9},
        ),
        (["--WITH_UNCERTAINTY_RECOMMENDATION"], {"action": "store_true"}),
        (["--WITH_CLUSTER_RECOMMENDATION"], {"action": "store_true"}),
        (["--WITH_SNUBA_LITE"], {"action": "store_true"}),
        (["--PLOT"], {"action": "store_true"}),
        (["--STOPPING_CRITERIA_UNCERTAINTY"], {"type": float, "default": 0.7}),
        (["--STOPPING_CRITERIA_ACC"], {"type": float, "default": 0.7}),
        (["--STOPPING_CRITERIA_STD"], {"type": float, "default": 0.7}),
        (
            ["--ALLOW_RECOMMENDATIONS_AFTER_STOP"],
            {"action": "store_true", "default": False},
        ),
        (["--DB_NAME_OR_TYPE"], {"default": "sqlite"}),
        (["--HYPER_SEARCH_TYPE"], {"default": "random"}),
        (["--USER_QUERY_BUDGET_LIMIT"], {"type": float, "default": 200}),
    ]
)

X_train, X_test, Y_train, Y_test, label_encoder_classes = get_dataset(
    config.DATASETS_PATH, config.DATASET_NAME
)

score = train_and_eval_dataset(
    config.DATASET_NAME,
    X_train,
    X_test,
    Y_train,
    Y_test,
    label_encoder_classes,
    hyper_parameters=vars(config),
)
print("Done with ", score)
