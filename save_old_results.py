from playhouse.postgres_ext import *
import operator
import pickle
import peewee

#  SELECT param_list_id, avg(fit_score), stddev(fit_score), avg(global_score), stddev(global_score), avg(start_set_size) as sss, count(*) FROM experimentresult WHERE start_set_size = 1 GROUP BY param_list_id ORDER BY 7 DESC, 4 DESC LIMIT 30;
from datetime import datetime, timedelta

#  1. import old experimentresult code, change name -> extract table, count "true weak stuff" from metrics_per_al_cycle["recommendations"]

db = peewee.DatabaseProxy()


class BaseModel(peewee.Model):
    class Meta:
        database = db


class ExperimentResult(BaseModel):
    class Meta:
        table_name = "experimentresult_paper_200_no_ground"

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
    uncertainty_recommendation_certainty_threshold = peewee.FloatField(null=True)
    uncertainty_recommendation_ratio = peewee.FloatField(null=True)
    snuba_lite_minimum_heuristic_accuracy = peewee.FloatField(null=True)
    cluster_recommendation_minimum_cluster_unity_size = peewee.FloatField(null=True)
    cluster_recommendation_ratio_labeled_unlabeled = peewee.FloatField(null=True)
    metrics_per_al_cycle = BinaryJSONField()  # json string
    amount_of_user_asked_queries = peewee.IntegerField(index=True)
    allow_recommendations_after_stop = peewee.BooleanField()
    stopping_criteria_uncertainty = peewee.FloatField()
    stopping_criteria_acc = peewee.FloatField()
    stopping_criteria_std = peewee.FloatField()

    # information of hyperparam run
    experiment_run_date = peewee.DateTimeField(default=datetime.now)
    fit_time = peewee.TextField()  # timedelta
    confusion_matrix_test = BinaryJSONField()  # json
    confusion_matrix_train = BinaryJSONField()  # json
    classification_report_train = BinaryJSONField()  # json
    classification_report_test = BinaryJSONField()  # json
    acc_train = peewee.FloatField(index=True)
    acc_test = peewee.FloatField(index=True)
    fit_score = peewee.FloatField(index=True)
    roc_auc = peewee.FloatField(index=True)

    global_score_with_weak_roc_auc_old = peewee.FloatField(index=True)
    global_score_with_weak_roc_auc_norm_old = peewee.FloatField(index=True)

    global_score_no_weak_roc_auc = peewee.FloatField(index=True, null=True)
    global_score_no_weak_acc = peewee.FloatField(index=True, null=True)
    global_score_with_weak_roc_auc = peewee.FloatField(index=True, null=True)
    global_score_with_weak_acc = peewee.FloatField(index=True, null=True)

    global_score_no_weak_roc_auc_norm = peewee.FloatField(index=True, null=True)
    global_score_no_weak_acc_norm = peewee.FloatField(index=True, null=True)
    global_score_with_weak_roc_auc_norm = peewee.FloatField(index=True, null=True)
    global_score_with_weak_acc_norm = peewee.FloatField(index=True, null=True)

    param_list_id = peewee.TextField(index=True)

    cv_fit_score_mean = peewee.FloatField(null=True)
    cv_fit_score_std = peewee.FloatField(null=True)

    thread_id = peewee.BigIntegerField(index=True)
    end_time = peewee.DateTimeField(index=True)


def get_db(db_name_or_type):
    # create Databases for storing the results
    if db_name_or_type == "sqlite":
        db = peewee.Sqlite
    elif db_name_or_type == "tunnel":
        db = PostgresqlExtDatabase(
            "jg", host="localhost", port=1111, password="test", user="jg"
        )
    else:
        db = PostgresqlExtDatabase(db_name_or_type)
    db.bind([ExperimentResult])
    db.create_tables([ExperimentResult])
    #  db.connect()

    return db


config = {
    "datasets_path": "../datasets",
    "db": "jg",
    "param_list_id": "best_global_score",
}

db = get_db(db_name_or_type=config["db"])

results = (
    ExperimentResult.select(
        ExperimentResult.param_list_id,
        #  ExperimentResult.acc_test_oracle,
        ExperimentResult.acc_test,
        ExperimentResult.fit_score,
        ExperimentResult.fit_time,
        #  ExperimentResult.amount_of_all_labels,
        ExperimentResult.amount_of_user_asked_queries,
        # ExperimentResult.classifier,
        # ExperimentResult.test_fraction,
        ExperimentResult.sampling,
        ExperimentResult.cluster,
        # ExperimentResult.nr_queries_per_iteration,
        ExperimentResult.with_uncertainty_recommendation,
        ExperimentResult.with_cluster_recommendation,
        ExperimentResult.uncertainty_recommendation_certainty_threshold,
        ExperimentResult.uncertainty_recommendation_ratio,
        ExperimentResult.cluster_recommendation_minimum_cluster_unity_size,
        ExperimentResult.cluster_recommendation_ratio_labeled_unlabeled,
        ExperimentResult.allow_recommendations_after_stop,
        ExperimentResult.experiment_run_date,
        ExperimentResult.metrics_per_al_cycle,
    ).where(
        (ExperimentResult.amount_of_user_asked_queries > 0)
        & (ExperimentResult.dataset_name == "dwtc")
        # & (ExperimentResult.experiment_run_date > (datetime(2020, 3, 24, 14, 0)))
        # & (ExperimentResult.experiment_run_date > (datetime(2020, 5, 8, 9, 20)))
        # & (ExperimentResult.with_cluster_recommendation == True)
        # & (ExperimentResult.with_uncertainty_recommendation == True)
        # & (peewee.fn.COUNT(ExperimentResult.id_field) == 3)
        # no stopping criterias
    )
    #  .order_by(ExperimentResult.acc_test.desc())
    .limit(10)
)

table = []
id = 0
for result in results:
    data = {**vars(result)["__data__"]}
    data["amount_of_all_labels"] = 200
    data["acc_test_oracle"] = data["acc_test"]

    data["weak?"] = operator.and_(
        operator.and_(
            operator.or_(
                data["with_uncertainty_recommendation"],
                data["with_cluster_recommendation"],
            ),
            data["amount_of_all_labels"] > 214,
        ),
        data["acc_test"] > data["acc_test_oracle"],
    )

    data["acc_test_all_better?"] = data["acc_test"] > data["acc_test_oracle"]
    data["true_weak?"] = operator.and_(
        operator.or_(
            data["with_uncertainty_recommendation"], data["with_cluster_recommendation"]
        ),
        data["amount_of_all_labels"] > 214,
    )
    data["interesting?"] = operator.and_(
        data["true_weak?"], data["acc_test_all_better?"]
    )
    table.append(data)
    id += 1

with open("old_results.pickle", "wb") as f:
    pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)
print(table)
