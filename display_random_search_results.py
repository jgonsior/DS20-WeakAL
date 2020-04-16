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
from collections import defaultdict, namedtuple
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
from tabulate import DataRow, TableFormat, _build_simple_row, tabulate

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
alt.themes.enable("opaque")
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
    ADDITIONAL_WHERE=True,
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
            & (ExperimentResult.stopping_criteria_acc == 1)
            & (ExperimentResult.stopping_criteria_std == 1)
            & (ExperimentResult.stopping_criteria_uncertainty == 1)
            & (ExperimentResult.dataset_name == DATASET)
            & ADDITIONAL_WHERE
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


def save_table_as_latex(table, destination, top=True):
    table = pd.DataFrame(table)
    if top:
        table["id"] = table["id"].apply(lambda x: "Top " + str(x + 1))
    table = table.set_index("id")

    numeric_column_names = table.select_dtypes(float).columns
    table[numeric_column_names] = table[numeric_column_names].applymap(
        "{0:2.2%}".format
    )

    # rename sapmling and cluster values
    table.sampling = table.sampling.str.replace("_", " ")
    table.sampling = table.sampling.str.replace("MostUncertain", "most uncertain")
    table.sampling = table.sampling.str.replace("lc", "least confident")
    table.cluster = table.cluster.str.replace("_", " ")
    table.cluster = table.cluster.str.replace("MostUncertain", "most uncertain")
    table.cluster = table.cluster.str.replace("lc", "least confident")
    table.cluster = table.cluster.str.replace("dummy", "single")

    # renamle column names
    table.columns = table.columns.str.replace("_", " ")
    table.columns = table.columns.str.replace("fit score", "end score")
    table.columns = table.columns.str.replace(
        "global score no weak acc", "global score"
    )
    table.columns = table.columns.str.replace(
        "amount of user asked queries", "# queries"
    )
    table.columns = table.columns.str.replace("acc test", "test accuracy")
    table.columns = table.columns.str.replace(
        "sampling", "sampling \\\\newline strategy"
    )
    table.columns = table.columns.str.replace("cluster", "cluster strategy")
    table.columns = table.columns.str.replace(
        "with uncertainty recommendation", "weak certainty?"
    )
    table.columns = table.columns.str.replace(
        "with cluster strategy recommendation", "weak cluster?"
    )

    table[["weak cluster?"]] = table[["weak cluster?"]].replace(
        [True, False], ["Yes", "No"]
    )
    table[["weak certainty?"]] = table[["weak certainty?"]].replace(
        {True: "Yes", False: "No",}
    )

    table = table.T

    def _latex_line_begin_tabular(colwidths, colaligns, booktabs=False):
        colwidths = [6.5] + [5 for _ in colwidths[1:]]
        colwidths = [11.5 * cw / sum(colwidths) for cw in colwidths]
        alignment = {"left": "R{1.5cm}", "right": "L", "center": "c", "decimal": "r"}
        #  tabular_columns_fmt = "".join([alignment.get(a, "l") for a in colaligns])
        tabular_columns_fmt = (
            "L{"
            + str(colwidths[0])
            + "cm}"
            + "".join(["R{" + str(cw) + "cm}" for cw in colwidths[1:]])
        )
        return "\n".join(
            [
                "\\begin{table}\\centering\\begin{tabularx}{\linewidth}{"
                + tabular_columns_fmt
                + "}",
                "\\toprule" if booktabs else "\\hline",
            ]
        )

    LATEX_ESCAPE_RULES = {
        r"&": r"\&",
        r"%": r"\%",
        r"$": r"\$",
        r"#": r"\#",
        r"_": r"\_",
        r"^": r"\^{}",
        r"{": r"\{",
        r"}": r"\}",
        r"~": r"\textasciitilde{}",
        #  "\\": r"\textbackslash{}",
        r"<": r"\ensuremath{<}",
        r">": r"\ensuremath{>}",
    }

    def _latex_row(cell_values, colwidths, colaligns, escrules=LATEX_ESCAPE_RULES):
        def escape_char(c):
            return escrules.get(c, c)

        escaped_values = ["".join(map(escape_char, cell)) for cell in cell_values]
        escaped_values = [
            "\\multicolumn{1}{r}{" + e + "}" if "%" in e else e for e in escaped_values
        ]
        escaped_values = [
            "\\multicolumn{1}{r}{" + e + "}" if e == "Yes" or e == "No" else e
            for e in escaped_values
        ]

        rowfmt = DataRow("", "&", "\\\\")
        return _build_simple_row(escaped_values, rowfmt)

    Line = namedtuple("Line", ["begin", "hline", "sep", "end"])
    my_latex_table = TableFormat(
        lineabove=partial(_latex_line_begin_tabular, booktabs=True),
        linebelowheader=Line("\\midrule", "", "", ""),
        linebetweenrows=None,
        linebelow=Line(
            "\\bottomrule\n\\end{tabularx}\\caption{\\tableCaption}\\end{table}",
            "",
            "",
            "",
        ),
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
    TOP_N,
    GROUP_SELECT,
    GROUP_SELECT_AGG,
    BUDGET,
    ORDER_BY,
    DATASET,
    ADDITIONAL_WHERE=True,
    LEGEND="",
):
    table = get_result_table(
        GROUP_SELECT=GROUP_SELECT,
        GROUP_SELECT_AGG=GROUP_SELECT_AGG,
        ADDITIONAL_SELECT=[],
        ORDER_BY=ORDER_BY,
        BUDGET=BUDGET,
        LIMIT=TOP_N + 1,
        ADDITIONAL_WHERE=ADDITIONAL_WHERE,
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


def compare_data(datasets, without_weak=True, dataset_name="dwtc", COLUMNS=3):
    charts = []

    alt.renderers.enable("html")
    all_data = pd.DataFrame()
    #  point_datas = pd.DataFrame()

    for dataset in datasets:
        for result in dataset:
            if dataset_name is not False:
                if result.dataset_name != dataset_name:
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
                    "top_n": result.legend.replace("_", "\_"),
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
                {"A": "Oracle", "C": "Weak", "U": "Weak", "G": "Oracle",}, inplace=True,
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
        if dataset_name != False:
            x_scale = alt.Scale()
        else:
            x_scale = alt.Scale(type="log")
    else:
        if dataset_name != False:
            x_scale = alt.Scale(domain=[0, 3000])
        else:
            x_scale = alt.Scale(type="log")
        show_top_legend = None
        show_thickness_legend = alt.Legend()

    lines = (
        alt.Chart(all_data,)
        .mark_trail(interpolate="step-before")
        .encode(
            x=alt.X("asked_queries:Q", title="\#Asked Queries", scale=x_scale),
            y=alt.Y(
                "test_acc:Q",
                title="Test Accuracy",
                scale=alt.Scale(domain=[0, 1], type="linear"),
            ),
            color=alt.Color("top_n:N", legend=show_top_legend,),
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
            padding=10,
            offset=5,
            #  labelSeparation=20,
            fillColor="#ffffff",
            gradientOpacity=0,
            #  labelOpacity=0,
            #  labelOverlap=True,
            title=None,
            columns=COLUMNS,
            #  strokeColor= "#878787"
        )
        .configure_axisBottom(labelSeparation=10)
        .properties(width=200, height=125)
        .configure_axisLeft(titlePadding=10)
        #  .properties(title="Comparison of ")
    )


def save_chart_as_latex(chart, base_title):
    save(
        chart, base_title + ".svg",
    )
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


def save_table_as_barchart(table, base_title, grouped="id", groupedTitle="Datasets"):
    df = pd.DataFrame(table)
    df.rename(
        columns={
            "fit_score": "end score",
            "global_score_no_weak_acc": "global score",
            "amount_of_user_asked_queries": "\% remaining budget",
            "acc_test": "test accuracy",
        },
        inplace=True,
    )

    alc = {
        "dwtc": 2889,
        "ibn_sina": 10361,
        "hiva": 21339,
        "orange": 25000,
        "sylva": 72626,
        "zebra": 30744,
    }

    df["\% remaining budget"] = df["\% remaining budget"].map(
        lambda q: 1 - q / config.BUDGET
    )
    newDf = pd.DataFrame(columns=["metric", "value", groupedTitle])
    # change df
    i = 0
    for index, row in df.iterrows():
        for metric in [
            "end score",
            "global score",
            "test accuracy",
            "\% remaining budget",
            "\% total asked oracle queries",
        ]:
            if groupedTitle != "Datasets" and metric == "\% total asked oracle queries":
                continue
            if metric == "\% remaining budget" and row[grouped] == "No Weak":
                row[metric] = 0
            if metric == "\% total asked oracle queries":
                value = row["\% remaining budget"] / alc[row[grouped]] * config.BUDGET
            else:
                value = row[metric]
            value = value * 100

            newDf.loc[i] = [
                metric,
                value,
                row[grouped].replace("_", " "),
            ]
            i += 1

    chart = (
        (
            alt.Chart(newDf)
            .mark_bar()
            .encode(
                x="metric",
                y=alt.Y(
                    "value",
                    axis=alt.Axis(format=".0r", title="Percentage",),
                    scale=alt.Scale(domain=[0, 100]),
                ),
                #  x=alt.X("metric:Q", title="dataset name"),
                #  y=alt.Y(), type="quantitative"),
                color=alt.Color("metric", title=None),
                column=groupedTitle,
            )
        )
        .properties(width=100, height=200)
        .configure_axisLeft(titlePadding=10)
        .configure_axisBottom(labelAngle=45, title=None, labels=False, ticks=False)
        .configure_legend(orient="bottom", columns=3, columnPadding=130)
    )

    #  save(chart, "test.png")
    save_chart_as_latex(chart, base_title)


if config.ACTION == "table":
    table = get_result_table(
        GROUP_SELECT=[ExperimentResult.param_list_id],
        GROUP_SELECT_AGG=[],
        ADDITIONAL_SELECT=[
            ExperimentResult.fit_score,
            ExperimentResult.global_score_no_weak_acc,
            ExperimentResult.amount_of_user_asked_queries,
            ExperimentResult.acc_test,
            #  ExperimentResult.classifier,
            #  ExperimentResult.test_fraction,
            ExperimentResult.sampling,
            ExperimentResult.cluster,
            #  ExperimentResult.nr_queries_per_iteration,
            ExperimentResult.with_uncertainty_recommendation,
            ExperimentResult.with_cluster_recommendation,
            #  ExperimentResult.uncertainty_recommendation_certainty_threshold,
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
        LIMIT=config.TOP,
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
                LEGEND="Top " + str(i + 1),
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
            stderr=subprocess.DEVNULL,
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


elif config.ACTION == "compare_rec":
    table = []
    for recommendations, name in zip(
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        ["No Weak", "Weak Certainty", "Weak Cluster", "Both",],
    ):
        if name == "No Weak":
            ORDER_BY = ExperimentResult.acc_test
            BUDGET = 500000
        else:
            BUDGET = config.BUDGET
            ORDER_BY = getattr(ExperimentResult, config.METRIC)
        table1 = get_result_table(
            GROUP_SELECT=[ExperimentResult.param_list_id],
            GROUP_SELECT_AGG=[],
            ADDITIONAL_SELECT=[
                ExperimentResult.fit_score,
                ExperimentResult.global_score_no_weak_acc,
                ExperimentResult.amount_of_user_asked_queries,
                ExperimentResult.acc_test,
                ExperimentResult.sampling,
                ExperimentResult.cluster,
                ExperimentResult.with_uncertainty_recommendation,
                ExperimentResult.with_cluster_recommendation,
                ExperimentResult.end_time,
            ],
            ORDER_BY=ORDER_BY,
            BUDGET=BUDGET,
            LIMIT=1,
            PARAM_LIST_ID=False,
            ADDITIONAL_WHERE=(
                (ExperimentResult.with_cluster_recommendation == recommendations[1])
                & (
                    ExperimentResult.with_uncertainty_recommendation
                    == recommendations[0]
                )
            ),
        )
        table1[0]["id"] = name
        if name == "No Weak":
            table1[0]["fit_score"] = 0
        table += table1
    save_table_as_latex(table, config.DESTINATION + ".tex", top=False)
    save_table_as_barchart(
        table,
        config.DESTINATION + "_barchart",
        grouped="id",
        groupedTitle="Used Weak Supervision Techniques",
    )

    datasets = []
    for recommendations, name in zip(
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        ["No Weak", "Weak Certainty", "Weak Cluster", "Both",],
    ):
        if name == "No Weak":
            ORDER_BY = ExperimentResult.acc_test
            BUDGET = 500000
        else:
            BUDGET = config.BUDGET
            ORDER_BY = getattr(ExperimentResult, config.METRIC)

        datasets.append(
            pre_fetch_data(
                0,
                GROUP_SELECT=[ExperimentResult.param_list_id],
                GROUP_SELECT_AGG=[],
                BUDGET=BUDGET,
                DATASET=config.DATASET,
                ORDER_BY=ORDER_BY,
                ADDITIONAL_WHERE=(
                    (ExperimentResult.with_cluster_recommendation == recommendations[1])
                    & (
                        ExperimentResult.with_uncertainty_recommendation
                        == recommendations[0]
                    )
                ),
                LEGEND=name,
            )
        )

    for with_or_without_weak in [True, False]:
        base_title = config.DESTINATION + "_" + str(with_or_without_weak)
        save(
            compare_data(datasets, with_or_without_weak, COLUMNS=1), base_title + ".svg"
        )
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

elif config.ACTION == "compare_all":
    table = []
    for dataset_name in ["dwtc", "ibn_sina", "hiva", "orange", "sylva", "zebra"]:
        table1 = get_result_table(
            GROUP_SELECT=[ExperimentResult.param_list_id],
            GROUP_SELECT_AGG=[],
            ADDITIONAL_SELECT=[
                ExperimentResult.fit_score,
                ExperimentResult.global_score_no_weak_acc,
                ExperimentResult.amount_of_user_asked_queries,
                ExperimentResult.acc_test,
                ExperimentResult.sampling,
                ExperimentResult.cluster,
                ExperimentResult.with_uncertainty_recommendation,
                ExperimentResult.with_cluster_recommendation,
            ],
            ORDER_BY=getattr(ExperimentResult, config.METRIC),
            BUDGET=config.BUDGET,
            LIMIT=1,
            PARAM_LIST_ID=False,
            DATASET=dataset_name,
        )
        table1[0]["id"] = dataset_name
        table += table1
    save_table_as_latex(table, config.DESTINATION + ".tex", top=False)
    save_table_as_barchart(table, config.DESTINATION + "_barchart")

    datasets = []
    for dataset_name in [
        "hiva",
        "dwtc",
        "ibn_sina",
        "hiva",
        "orange",
        "sylva",
        "zebra",
    ]:
        datasets.append(
            pre_fetch_data(
                0,
                GROUP_SELECT=[ExperimentResult.param_list_id],
                GROUP_SELECT_AGG=[ExperimentResult.acc_test,],
                BUDGET=config.BUDGET,
                ORDER_BY=getattr(ExperimentResult, config.METRIC),
                DATASET=dataset_name,
                LEGEND=dataset_name,
            )
        )

    for with_or_without_weak in [True, False]:
        base_title = config.DESTINATION + "_" + str(with_or_without_weak)
        save(
            compare_data(datasets, with_or_without_weak, dataset_name=False, COLUMNS=2),
            base_title + ".svg",
        )
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

elif config.ACTION == "budgets":
    results = (
        ExperimentResult.select(
            ExperimentResult.amount_of_user_asked_queries,
            peewee.fn.MAX(ExperimentResult.acc_test).alias("max"),
            #  ExperimentResult.acc_test.alias("max"),
        )
        .where(
            (ExperimentResult.stopping_criteria_acc == 1)
            & (ExperimentResult.stopping_criteria_std == 1)
            & (ExperimentResult.stopping_criteria_uncertainty == 1)
            & (ExperimentResult.dataset_name == "dwtc")
            & (ExperimentResult.sampling != "random")
            #  & (ExperimentResult.cluster != "random")
            #  & (ExperimentResult.with_cluster_recommendation == True)
            #  & (ExperimentResult.with_uncertainty_recommendation == True)
        )
        .group_by(ExperimentResult.amount_of_user_asked_queries)
        .order_by(
            ExperimentResult.amount_of_user_asked_queries.desc(),
            #  ExperimentResult.acc_test.desc()
            #  peewee.fn.MAX(ExperimentResult.acc_test)
            #  peewee.fn.COUNT(ExperimentResult.).desc(),
            #  peewee.fn.AVG(ORDER_BY).desc(),
        )
        #  .limit(10)
    )
    data = []
    for result in results:
        data.append((result.amount_of_user_asked_queries, result.max))
        #  print("{}\t{}".format(result.amount_of_user_asked_queries, result.max))
    df = pd.DataFrame(data)
    df.columns = ["budget", "end test accuracy"]
    print(df)
    #  import numpy as np

    #  x = np.arange(100)
    #  df = pd.DataFrame({"budget": x, "end test accuracy": np.sin(x / 5)})
    alt.data_transformers.disable_max_rows()
    chart = (
        alt.Chart(df)
        .mark_circle(opacity=0.3, color="orange")
        .encode(x=alt.X("budget", title="Budget"), y="end test accuracy")
    ).properties(width=500, height=200)
    chart = chart + chart.transform_loess("budget", "end test accuracy").mark_line(
        #  color="lightblue"
    )

    base_title = config.DESTINATION
    save(chart, base_title + ".svg")

    subprocess.run(
        "inkscape -D -z --file "
        + base_title
        + ".svg --export-pdf "
        + base_title
        + ".pdf --export-latex",
        shell=True,
        stderr=subprocess.DEVNULL,
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
