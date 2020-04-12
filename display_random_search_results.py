import argparse
import contextlib
import datetime
import fileinput
import io
import logging
import multiprocessing
import os
import random
import subprocess
import sys
from collections import namedtuple
from datetime import datetime, timedelta
from functools import partial
from itertools import chain, combinations
from pprint import pprint
from timeit import default_timer as timer

import altair as alt
import altair_viewer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peewee
from altair_saver import save
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from IPython.core.display import HTML, display
from json_tricks import dumps, loads
from playhouse.shortcuts import model_to_dict
from scipy.stats import randint, uniform
from sklearn.datasets import load_iris
from tabulate import TableFormat, _latex_row, tabulate

from active_learning.cluster_strategies import (
    DummyClusterStrategy,
    MostUncertainClusterStrategy,
    RandomClusterStrategy,
    RoundRobinClusterStrategy,
)
from active_learning.dataStorage import DataStorage
from active_learning.experiment_setup_lib import (
    ExperimentResult,
    classification_report_and_confusion_matrix,
    get_db,
    get_single_al_run_stats_row,
    get_single_al_run_stats_table_header,
    load_and_prepare_X_and_Y,
    standard_config,
)
from active_learning.sampling_strategies import (
    BoundaryPairSampler,
    CommitteeSampler,
    RandomSampler,
    UncertaintySampler,
)

#  alt.renderers.enable("altair_viewer")
#  alt.renderers.enable('vegascope')

config = standard_config(
    [
        (["--ACTION"], {}),
        (["--TOP"], {"type": int}),
        (["--TOP2"], {"type": int}),
        (["--TOP3"], {"type": int}),
        (["--BUDGET"], {"type": int}),
        (["--DATASET"], {}),
        (["--METRIC"], {}),
        (["--DESTINATION"], {}),
        (["--RANDOM_SEED"], {"type": int, "default": -1}),
        (["--LOG_FILE"], {"default": "log.txt"}),
        (["--DB"], {"default": "tunnel"}),
    ],
    False,
)

db = get_db(db_name_or_type=config.DB)


# select count(*), dataset_name from experimentresult group by dataset_name;
results = ExperimentResult.select(
    ExperimentResult.dataset_name,
    peewee.fn.COUNT(ExperimentResult.id_field).alias("dataset_name_count"),
).group_by(ExperimentResult.dataset_name)

for result in results:
    print("{:>4,d} {}".format(result.dataset_name_count, result.dataset_name))

# & (ExperimentResult.experiment_run_date > (datetime(2020, 3, 24, 14, 0))) # no stopping criterias
#  & (ExperimentResult.experiment_run_date > (datetime(2020, 3, 30, 12, 23))) # optics


def get_result_table(
    GROUP_SELECT=[ExperimentResult.param_list_id],
    GROUP_SELECT_AGG=[
        ExperimentResult.fit_score,
        ExperimentResult.global_score_no_weak_acc,
        ExperimentResult.amount_of_user_asked_queries,
    ],
    ADDITIONAL_SELECT=[
        ExperimentResult.classifier,
        ExperimentResult.test_fraction,
        ExperimentResult.sampling,
        ExperimentResult.cluster,
        ExperimentResult.nr_queries_per_iteration,
        ExperimentResult.with_uncertainty_recommendation,
        ExperimentResult.with_cluster_recommendation,
        ExperimentResult.uncertainty_recommendation_certainty_threshold,
        ExperimentResult.uncertainty_recommendation_ratio,
        ExperimentResult.cluster_recommendation_minimum_cluster_unity_size,
        ExperimentResult.cluster_recommendation_ratio_labeled_unlabeled,
        ExperimentResult.allow_recommendations_after_stop,
        ExperimentResult.stopping_criteria_uncertainty,
        ExperimentResult.stopping_criteria_acc,
        ExperimentResult.stopping_criteria_std,
        ExperimentResult.experiment_run_date,
    ],
    ORDER_BY=ExperimentResult.global_score_no_weak_acc,
    BUDGET=2000,
    LIMIT=20,
    DATASET="dwtc",
    PARAM_LIST_ID=True,
):
    results = (
        ExperimentResult.select(
            *GROUP_SELECT,
            *[
                f(s)
                for s in GROUP_SELECT_AGG
                for f in (
                    lambda s: peewee.fn.AVG(s).alias("avg_" + s.name),
                    lambda s: peewee.fn.STDDEV(s).alias("stddev_" + s.name),
                )
            ]
        )
        .where(
            (ExperimentResult.amount_of_user_asked_queries < BUDGET)
            & (ExperimentResult.experiment_run_date > (datetime(2020, 3, 24, 14, 0)))
            & (ExperimentResult.dataset_name == DATASET)
        )
        .group_by(ExperimentResult.param_list_id)
        .order_by(
            peewee.fn.COUNT(ExperimentResult.id_field).desc(),
            peewee.fn.AVG(ORDER_BY).desc(),
        )
        .limit(LIMIT)
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
            ExperimentResult.select(*ADDITIONAL_SELECT)
            .where(
                (ExperimentResult.param_list_id == data["param_list_id"])
                & (ExperimentResult.dataset_name == DATASET)
            )
            .limit(1)
        )[0]

        data = {**data, **vars(one_param_list_id_result)["__data__"]}

        if not PARAM_LIST_ID:
            del data["param_list_id"]
        table.append(data)
        id += 1
    return table


def save_table_as_latex(table, destination):
    table = pd.DataFrame(table)
    table["id"] = table["id"].apply(lambda x: "Top " + str(x))
    table = table.set_index("id")

    numeric_column_names = table.select_dtypes(float).columns
    table[numeric_column_names] = table[numeric_column_names].applymap(
        "{0:2.2%}".format
    )

    # renamle column names
    table.columns = table.columns.str.replace("_", " ")
    # rename sapmling and cluster values
    table.sampling = table.sampling.str.replace("_", " ")

    table = table.T

    def _latex_line_begin_tabular(colwidths, colaligns, booktabs=False):
        alignment = {"left": "p{3cm}", "right": "r", "center": "c", "decimal": "r"}
        tabular_columns_fmt = "".join([alignment.get(a, "l") for a in colaligns])
        return "\n".join(
            [
                "\\begin{tabularx}{\linewidth}{" + tabular_columns_fmt + "}",
                "\\toprule" if booktabs else "\\hline",
            ]
        )

    Line = namedtuple("Line", ["begin", "hline", "sep", "end"])
    my_latex_table = TableFormat(
        lineabove=partial(_latex_line_begin_tabular, booktabs=True),
        linebelowheader=Line("\\midrule", "", "", ""),
        linebetweenrows=None,
        linebelow=Line("\\bottomrule\n\\end{tabularx}", "", "", ""),
        headerrow=_latex_row,
        datarow=_latex_row,
        padding=1,
        with_header_hide=None,
    )

    with open(destination, "w") as f:
        f.write(tabulate(table, headers="keys", tablefmt=my_latex_table))


def display_table(original_table, transpose=True):
    df = pd.DataFrame(original_table)
    if transpose:
        df = df.T

    print(tabulate(df, headers="keys", floatfmt=".2f"))


def pre_fetch_data(
    TOP_N, GROUP_SELECT, GROUP_SELECT_AGG, BUDGET, ORDER_BY, DATASET, LEGEND=""
):
    table = get_result_table(
        GROUP_SELECT=GROUP_SELECT,
        GROUP_SELECT_AGG=GROUP_SELECT_AGG,
        ADDITIONAL_SELECT=[],
        ORDER_BY=ORDER_BY,
        BUDGET=BUDGET,
        LIMIT=TOP_N + 1,
        PARAM_LIST_ID=True,
    )

    best_param_list_id = table[TOP_N]["param_list_id"]

    results = ExperimentResult.select().where(
        (ExperimentResult.param_list_id == best_param_list_id)
        & (ExperimentResult.dataset_name == DATASET)
    )

    loaded_data = []
    for result in results:
        setattr(result, "legend", LEGEND)
        loaded_data.append(result)
    print("Loaded Top " + str(TOP_N) + " data")

    return loaded_data


def visualise_top_n(data):
    charts = []

    alt.renderers.enable("html")

    for result in data:
        metrics = loads(result.metrics_per_al_cycle)
        test_data_metrics = [
            metrics["test_data_metrics"][0][f][0]["weighted avg"]
            for f in range(0, len(metrics["test_data_metrics"][0]))
        ]
        test_acc = [
            metrics["test_data_metrics"][0][f][0]["accuracy"]
            for f in range(0, len(metrics["test_data_metrics"][0]))
        ]

        data = pd.DataFrame(
            {
                "iteration": range(0, len(metrics["all_unlabeled_roc_auc_scores"])),
                "all_unlabeled_roc_auc_scores": metrics["all_unlabeled_roc_auc_scores"],
                "query_length": metrics["query_length"],
                "recommendation": metrics["recommendation"],
                "query_strong_accuracy_list": metrics["query_strong_accuracy_list"],
                "f1": [i["f1-score"] for i in test_data_metrics],
                "test_acc": test_acc,
                #'asked_queries': [sum(metrics['query_length'][:i]) for i in range(0, len(metrics['query_length']))],
            }
        )

        # bar width
        data["asked_queries"] = data["query_length"].cumsum()
        data["asked_queries_end"] = data["asked_queries"].shift(fill_value=0)

        # print(data[['asked_queries', 'query_length']])

        data["recommendation"] = data["recommendation"].replace(
            {
                "A": "Oracle",
                "C": "Weak Cluster",
                "U": "Weak Certainty",
                "G": "Ground Truth",
            }
        )

        # data = data[:100]

        # calculate global score OHNE

        chart = (
            alt.Chart(data)
            .mark_rect(
                # point=True,
                # line=True,
                # interpolate='step-after',
            )
            .encode(
                x=alt.X("asked_queries_end", title="#asked queries (weak and oracle)"),
                x2="asked_queries",
                color=alt.Color("recommendation", scale=alt.Scale(scheme="tableau10")),
                tooltip=[
                    "iteration",
                    "f1",
                    "test_acc",
                    "all_unlabeled_roc_auc_scores",
                    "query_strong_accuracy_list",
                    "query_length",
                    "recommendation",
                ],
                # scale=alt.Scale(domain=[0,1])
            )
            .properties(title=result.dataset_name)
            .interactive()
        )
        charts.append(
            alt.hconcat(
                chart.encode(
                    alt.Y(
                        "all_unlabeled_roc_auc_scores", scale=alt.Scale(domain=[0, 1])
                    )
                ).properties(title=result.dataset_name + ": roc_auc"),
                # chart.encode(alt.Y('f1', scale=alt.Scale(domain=[0,1]))).properties(title=result.dataset_name + ': f1'),
                chart.encode(
                    alt.Y("test_acc", scale=alt.Scale(domain=[0, 1]))
                ).properties(title=result.dataset_name + ": test_acc"),
            )
        )

    return alt.vconcat(*charts).configure()


def compare_data(datasets, without_weak=True):
    charts = []

    alt.renderers.enable("html")
    all_data = pd.DataFrame()
    #  point_datas = pd.DataFrame()

    for dataset in datasets:
        for result in dataset:
            if result.dataset_name != "dwtc":
                continue
            metrics = loads(result.metrics_per_al_cycle)
            test_data_metrics = [
                metrics["test_data_metrics"][0][f][0]["weighted avg"]
                for f in range(0, len(metrics["test_data_metrics"][0]))
            ]
            test_acc = [
                metrics["test_data_metrics"][0][f][0]["accuracy"]
                for f in range(0, len(metrics["test_data_metrics"][0]))
            ]

            data = pd.DataFrame(
                {
                    "iteration": range(0, len(metrics["all_unlabeled_roc_auc_scores"])),
                    "all_unlabeled_roc_auc_scores": metrics[
                        "all_unlabeled_roc_auc_scores"
                    ],
                    "query_length": metrics["query_length"],
                    "recommendation": metrics["recommendation"],
                    "query_strong_accuracy_list": metrics["query_strong_accuracy_list"],
                    "f1": [i["f1-score"] for i in test_data_metrics],
                    "test_acc": test_acc,
                    "top_n": result.legend,
                    "color": 4,
                    "opacity": 0.7,
                    "size": metrics["recommendation"]
                    #'asked_queries': [sum(metrics['query_length'][:i]) for i in range(0, len(metrics['query_length']))],
                }
            )

            if without_weak:
                data = pd.concat(
                    [data[data.recommendation == "G"], data[data.recommendation == "A"]]
                )

            # bar width
            data["asked_queries"] = data["query_length"].cumsum()
            data["asked_queries_end"] = data["asked_queries"].shift(fill_value=0)

            data["recommendation"].replace(
                {
                    "A": "Oracle",
                    "C": "Weak Cluster",
                    "U": "Weak Certainty",
                    "G": "Ground Truth",
                },
                inplace=True,
            )

            data["size"].replace(
                {
                    "A": "Oracle",
                    "C": "Recommendation",
                    "U": "Recommendation",
                    "G": "Oracle",
                },
                inplace=True,
            )

            #  if not without_weak:
            #  point_data = data[data.recommendation != "Oracle"]
            #  print(data)

            all_data = pd.concat([all_data, data])
            #  if not without_weak:
            #  point_datas = pd.concat([point_datas, point_data])

    #  points = (
    #  alt.Chart(point_datas)
    #  .mark_point()
    #  .encode(
    #  x="asked_queries:Q",
    #  y="test_acc:Q",
    #  shape="recommendation:N",
    #  #  color="color:N",
    #  #  color="recommendation:N",
    #  )
    #  )

    if without_weak:
        show_top_legend = alt.Legend()
        show_thickness_legend = None
    else:
        show_top_legend = None
        show_thickness_legend = alt.Legend()

    lines = (
        alt.Chart(all_data,)
        .mark_trail(interpolate="step-before")
        .encode(
            x=alt.X(
                "asked_queries:Q",
                title="Asked Queries",
                scale=alt.Scale(domain=[0, 2900], type="linear"),
            ),
            y=alt.Y(
                "test_acc:Q",
                title="Test Accuracy",
                scale=alt.Scale(domain=[0, 1], type="linear"),
            ),
            color=alt.Color("top_n:N", legend=show_top_legend),
            opacity=alt.Opacity("opacity", legend=None),
            size=alt.Size("size:N", legend=show_thickness_legend)
            # shape="top_n",
            # strokeDash="top_n",
            # shape="recommendation",
            # color="recommendation:N",
        )
    )
    plot = lines
    return (
        alt.layer(plot)
        .resolve_scale(opacity="independent", color="independent", shape="independent")
        .configure_legend(
            orient="bottom-right",
            #  padding=10,
            #  fillColor="#f1f1f1",
            labelOverlap=True,
            title=None,
        )
        .properties(width=200, height=200)
        #  .properties(title="Comparison of ")
    )


if config.ACTION == "table":
    table = get_result_table(
        GROUP_SELECT=[ExperimentResult.param_list_id],
        GROUP_SELECT_AGG=[],
        ADDITIONAL_SELECT=[
            ExperimentResult.fit_score,
            ExperimentResult.global_score_no_weak_acc,
            ExperimentResult.amount_of_user_asked_queries,
            #  ExperimentResult.classifier,
            #  ExperimentResult.test_fraction,
            ExperimentResult.sampling,
            ExperimentResult.cluster,
            #  ExperimentResult.nr_queries_per_iteration,
            ExperimentResult.with_uncertainty_recommendation,
            ExperimentResult.with_cluster_recommendation,
            ExperimentResult.uncertainty_recommendation_certainty_threshold,
            #  ExperimentResult.uncertainty_recommendation_ratio,
            #  ExperimentResult.cluster_recommendation_minimum_cluster_unity_size,
            #  ExperimentResult.cluster_recommendation_ratio_labeled_unlabeled,
            #  ExperimentResult.allow_recommendations_after_stop,
            #  ExperimentResult.stopping_criteria_uncertainty,
            #  ExperimentResult.stopping_criteria_acc,
            #  ExperimentResult.stopping_criteria_std,
            #  ExperimentResult.experiment_run_date,
        ],
        ORDER_BY=getattr(ExperimentResult, config.METRIC),
        BUDGET=config.BUDGET,
        LIMIT=3,
        PARAM_LIST_ID=False,
    )
    save_table_as_latex(table, config.DESTINATION + ".tex")

    datasets = []
    for i in range(0, config.TOP):
        datasets.append(
            pre_fetch_data(
                i,
                GROUP_SELECT=[ExperimentResult.param_list_id],
                GROUP_SELECT_AGG=[],
                BUDGET=config.BUDGET,
                DATASET=config.DATASET,
                ORDER_BY=getattr(ExperimentResult, config.METRIC),
                LEGEND="Top " + str(i),
            )
        )

    for with_or_without_weak in [True, False]:
        base_title = config.DESTINATION + "_" + str(with_or_without_weak)
        save(compare_data(datasets, with_or_without_weak), base_title + ".svg")
        subprocess.run(
            "inkscape -D -z --file "
            + base_title
            + ".svg --export-pdf "
            + base_title
            + ".pdf --export-latex",
            shell=True,
        )
        with fileinput.FileInput(
            base_title + ".pdf_tex", inplace=True, backup=".bak"
        ) as file:
            for line in file:
                print(
                    line.replace(
                        base_title.split("/")[-1] + ".pdf",
                        "results/" + base_title.split("/")[-1] + ".pdf",
                    ),
                    end="",
                )

elif config.ACTION == "plot":
    loaded_data = pre_fetch_data(
        config.TOP,
        GROUP_SELECT=[ExperimentResult.param_list_id],
        GROUP_SELECT_AGG=[],
        BUDGET=config.BUDGET,
        DATASET=config.DATASET,
        ORDER_BY=getattr(ExperimentResult, config.METRIC),
    )

    save(visualise_top_n(loaded_data), config.DESTINATION)
