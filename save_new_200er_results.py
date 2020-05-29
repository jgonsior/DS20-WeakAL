from active_learning.experiment_setup_lib import ExperimentResult, get_db
import operator
import pickle

#  SELECT param_list_id, avg(fit_score), stddev(fit_score), avg(global_score), stddev(global_score), avg(start_set_size) as sss, count(*) FROM experimentresult WHERE start_set_size = 1 GROUP BY param_list_id ORDER BY 7 DESC, 4 DESC LIMIT 30;
from datetime import datetime, timedelta

config = {
    "datasets_path": "../datasets",
    "db": "jg",
    "param_list_id": "best_global_score",
}

db = get_db(db_name_or_type=config["db"])

results = (
    ExperimentResult.select(ExperimentResult.param_list_id,)
    .where(
        (ExperimentResult.amount_of_user_asked_queries < 211)
        & (ExperimentResult.dataset_name == "dwtc")
        # & (ExperimentResult.experiment_run_date > (datetime(2020, 3, 24, 14, 0)))
        # & (ExperimentResult.experiment_run_date > (datetime(2020, 5, 8, 9, 20)))
        # & (ExperimentResult.with_cluster_recommendation == True)
        # & (ExperimentResult.with_uncertainty_recommendation == True)
        # & (peewee.fn.COUNT(ExperimentResult.id_field) == 3)
        # no stopping criterias
    )
    .order_by(ExperimentResult.acc_test.desc())
    .limit(1000000)
)

table = []
id = 0
for result in results:
    data = {**{"id": id}, **vars(result)}
    data["param_list_id"] = data["__data__"]["param_list_id"]
    del data["__data__"]
    del data["_dirty"]
    del data["__rel__"]

    # get one param_list_id

    one_param_list_id_result = (
        ExperimentResult.select(
            ExperimentResult.acc_test_oracle,
            ExperimentResult.acc_test,
            ExperimentResult.fit_score,
            ExperimentResult.fit_time,
            ExperimentResult.amount_of_all_labels,
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
        )
        .where(ExperimentResult.param_list_id == data["param_list_id"])
        .limit(1)
    )[0]
    data["weak?"] = operator.and_(
        operator.and_(
            operator.or_(
                one_param_list_id_result.with_uncertainty_recommendation,
                one_param_list_id_result.with_cluster_recommendation,
            ),
            one_param_list_id_result.amount_of_all_labels > 214,
        ),
        one_param_list_id_result.acc_test > one_param_list_id_result.acc_test_oracle,
    )

    data["acc_test_all_better?"] = (
        one_param_list_id_result.acc_test > one_param_list_id_result.acc_test_oracle
    )
    data["true_weak?"] = operator.and_(
        operator.or_(
            one_param_list_id_result.with_uncertainty_recommendation,
            one_param_list_id_result.with_cluster_recommendation,
        ),
        one_param_list_id_result.amount_of_all_labels > 214,
    )
    data["interesting?"] = operator.and_(
        data["true_weak?"], data["acc_test_all_better?"]
    )
    data = {**data, **vars(one_param_list_id_result)["__data__"]}

    table.append(data)
    id += 1

with open("200er_results.pickle", "wb") as f:
    pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)
