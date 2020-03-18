import argparse
import glob
import os
import pickle
import sys
from abc import abstractmethod
from collections import defaultdict
from math import sqrt
from pprint import pprint

import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

import functions

parser = argparse.ArgumentParser()
parser.add_argument("--strategy")
parser.add_argument("--output")
parser.add_argument("--query", action="store_true")
parser.add_argument("--wishedPlots", default="multiple")
parser.add_argument("--stopping", action="store_true")
parser.add_argument("--stop_uncertainty", type=float)
parser.add_argument("--stop_acc", type=float)
parser.add_argument("--stop_std", type=float)

store_location = "/home/julius/coding/dbs_svn2/seadata-active_learning/fig"

config = parser.parse_args()

SPINE_COLOR = "black"
plt.style.use("seaborn-paper")
#  rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
#  rc('text', usetex=True)

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()

#  if not os.path.exists(config.output):
#  os.makedirs(config.output)


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert columns in [1, 2]

    if fig_width is None:
        fig_width = 4.5 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        #  fig_height = fig_width * golden_mean  # height in inches
        fig_height = 3.8 * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print(
            "WARNING: fig_height too large:"
            + fig_height
            + "so will reduce to"
            + MAX_HEIGHT_INCHES
            + "inches."
        )
        fig_height = MAX_HEIGHT_INCHES

    size = 10
    params = {
        "backend": "ps",
        "text.latex.preamble": ["\\usepackage{gensymb}"],
        "axes.labelsize": size,  # fontsize for x and y labels (was 10)
        "axes.titlesize": size,
        "font.size": size,  # was 10
        "legend.fontsize": 7,  # was 10
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        #  'text.usetex': True,
        "figure.figsize": [fig_width, fig_height],
        "font.family": "serif",
    }

    matplotlib.rcParams.update(params)


def format_axes(ax, right=False):

    if not right:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    else:
        ax.spines["top"].set_visible(False)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position("bottom")
    #  ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction="out", color=SPINE_COLOR)

    return ax


def plot_metrics(
    metric_list,
    label_list,
    title,
    passive_accuracy=None,
    markers_list=None,
    legend=True,
    legend_font_size=7,
    second_axis=None,
    xlabel="# Queried Sheets",
    ylabel="Accuracy",
    highestFirst=False,
    legendLoc=0,
):
    if markers_list is None:
        markers_list = [None for i in metric_list]

    if second_axis is not None:
        #  plt.yticks(np.arange(0, 1.0, step=0.1))
        ax1 = plt.gca()
        #  ax1.set_yticks(np.arange(0.75, 0.9, 0.05))
        #  ax1.ylim([0.75, 0.9])
        ax2 = plt.gca().twinx()
        ax2.set_ylabel(second_axis)
        #  ax2.set_ylim(ax1.get_ylim())
        #  ax1.tick_params(axis='y')
        #  ax1.set_ylim([0.75, 0.9])
    else:
        ax1 = plt.gca()
    if passive_accuracy is not None:
        ax1.axhline(
            y=passive_accuracy,
            color="black",
            label="Baseline",
            linestyle="dotted",
            linewidth=0.4,
        )

    if highestFirst == True:
        highestAcc = np.argmax(metric_list[0])
        pprint(highestAcc)
        plt.axvline(x=highestAcc, color="purple", linestyle="dashed", linewidth=0.4)
        #  plt.annotate("max acc", (highestAcc, 1), fontsize=8)

    for metrics, label, marker in zip(metric_list, label_list, markers_list):
        if marker is None or isinstance(marker, float):
            markevery = None
            markersize = 0.4
        else:
            markevery = [marker]
            markersize = 5.5

        if second_axis is not None:
            if metrics == metric_list[0]:
                ax = ax1
                color = "purple"
            else:
                color = None
                ax = ax2
        else:
            color = None
            ax = plt.gca()

        ax.plot(
            metrics,
            label=label,
            linewidth=0.4,
            marker="o",
            color=color,
            markersize=markersize,
            markevery=markevery,
        )

    plt.xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    if legend:
        if second_axis is not None:
            plt.legend(prop={"size": legend_font_size})
            legend1 = ax1.legend(loc=6)
            ax2.legend(loc=4)
            legend1.remove()
            plt.gca().add_artist(legend1)
        else:
            plt.legend(loc=legendLoc)
    plt.title(title)

    format_axes(ax1, True)
    if second_axis is not None:
        format_axes(ax2, True)
    plt.tight_layout()

    return plt


def plot_accs_query(strategy, start_size, passive_accuracy=None):
    test_accs = []
    train_accs = []
    query_accs = []

    for test_data, train_data, query_data in zip(
        metrics_per_time["test_data"][0],
        metrics_per_time["train_data"][0],
        metrics_per_time["query_data"][0],
    ):
        test_accs.append(test_data[0]["accuracy"])
        train_accs.append(train_data[0]["accuracy"])
        query_accs.append(query_data[0]["accuracy"])

    return plot_metrics(
        [test_accs, train_accs, query_accs],
        ["Test", "Training", "Query"],
        strategy + ": " + start_size,
        passive_accuracy=passive_accuracy,
    )


def plot_accs(strategy, start_size, passive_accuracy=None):
    test_accs, train_accs = get_metrics(metrics_per_time)
    return plot_metrics(
        [test_accs, train_accs],
        ["Test", "Training"],
        strategy + ": " + start_size,
        passive_accuracy=passive_accuracy,
    )


def get_metrics(metrics_per_time):
    test_accs = []
    train_accs = []

    for test_data, train_data in zip(
        metrics_per_time["test_data"][0], metrics_per_time["train_data"][0]
    ):
        test_accs.append(test_data[0]["accuracy"])
        train_accs.append(train_data[0]["accuracy"])

    #  return train_accs, test_accs
    return test_accs, train_accs


def plot_classes(strategy, start_size, passive_accuracy=None):
    labels = set(metrics_per_time["test_data"][0][0][0].keys())
    labels = labels.difference(set(["accuracy", "macro avg", "weighted avg"]))
    test_accs = {}
    train_accs = {}

    for label in labels:
        test_accs[label] = []
        train_accs[label] = []

    for test_data, train_data in zip(
        metrics_per_time["test_data"][0], metrics_per_time["train_data"][0]
    ):
        for label in labels:
            test_accs[label].append(test_data[0][label]["f1-score"])
            train_accs[label].append(train_data[0][label]["f1-score"])

    metrics = list(test_accs.values()) + list(train_accs.values())
    plot_labels = ["Test " + label for label in labels] + [
        "Train " + label for label in labels
    ]

    return plot_metrics(
        metrics,
        plot_labels,
        strategy + ": " + start_size,
        passive_accuracy=passive_accuracy,
    )


def plot_distributions_of_asked_queries(strategy, start_size):
    labels = set([label for label in metrics_per_time["test_data"][0][0][0].keys()])
    labels = labels.difference(set(["accuracy", "macro avg", "weighted avg"]))
    distributions = {}

    for label in labels:
        distributions[label] = []

    for query_set_distribution in metrics_per_time["query_set_distribution"][0]:
        for label in labels:
            distributions[label].append(query_set_distribution[label])

    ys = []
    for i, label in enumerate(labels):
        ys.append(distributions[label])
        #  plt.plot(distributions[label], label=label)

    plt.stackplot(range(len(ys[0])), ys, labels=labels)
    plt.xlabel("Learning Iterations (150 samples each)")
    plt.ylabel("Distribution of asked queries")

    plt.legend()
    plt.title(strategy + ": " + start_size)

    return plt


def get_passive_accuracy(start_size, batch_size):
    # get classification results
    with open(
        config.output
        + "/active_"
        + config.strategy
        + "_start_"
        + str(start_size)
        + "_"
        + str(batch_size)
        + ".txt",
        "r",
    ) as results_file:

        passive_accuracy = 0

        for line in results_file:
            if (
                "accuracy of simple classifier training \033[1m biggest possible training set"
                in line
            ):
                passive_accuracy = float(line.split(" ")[-1][:-1])

        return passive_accuracy


if config.wishedPlots == "pages":
    with PdfPages(config.output + "/" + config.strategy + "_acc_full.pdf") as acc_pdf:
        with PdfPages(
            config.output + "/" + config.strategy + "_dist_full.pdf"
        ) as dist_pdf:
            with PdfPages(
                config.output + "/" + config.strategy + "_classes_full.pdf"
            ) as classes_pdf:

                for pickle_file in sorted(
                    glob.glob(config.output + "/" + config.strategy + "_*.pickle")
                ):
                    print(pickle_file)
                    with open(pickle_file, "rb") as f:
                        metrics_per_time = pickle.load(f)

                        # don't ask
                        start_size = pickle_file.split("/")[2].split("_")[-2]
                        batch_size = (
                            pickle_file.split("/")[2].split("_")[-1].split(".")[0]
                        )

                        passive_accuracy = get_passive_accuracy(start_size, batch_size)

                        classes_plt = plot_classes(
                            strategy=config.strategy, start_size=start_size
                        )

                        classes_pdf.savefig()
                        classes_plt.close()

                        if config.query:
                            acc_plt = plot_accs_query(
                                strategy=config.strategy,
                                start_size=start_size,
                                passive_accuracy=passive_accuracy,
                            )
                        else:
                            acc_plt = plot_accs(
                                strategy=config.strategy,
                                start_size=start_size,
                                passive_accuracy=passive_accuracy,
                            )
                        acc_pdf.savefig()
                        acc_plt.close()

                        dist_plt = plot_distributions_of_asked_queries(
                            strategy=config.strategy, start_size=start_size
                        )

                        dist_pdf.savefig()
                        dist_plt.close()
elif config.wishedPlots == "start_set_sizes":
    latexify(columns=1)
    metrics_list = []
    label_list = []
    batch_size = 150
    for start_size in [0.01, 0.05, 0.1]:
        start_size = str(start_size)
        pickle_file = (
            config.output
            + "/"
            + config.strategy
            + "_"
            + start_size
            + "_"
            + str(batch_size)
            + ".pickle"
        )
        print(pickle_file)
        with open(pickle_file, "rb") as f:
            metrics_per_time = pickle.load(f)
        passive_accuracy = get_passive_accuracy(start_size, batch_size)

        test_accs, train_accs = get_metrics(metrics_per_time)

        metrics_list.append(test_accs)
        #  metrics_list.append(train_accs)
        label_list.append("{0:.0%}".format(float(start_size)))
        #  label_list.append(start_size + " (train)")

    plot = plot_metrics(
        metrics_list,
        label_list,
        "",
        #  legend=False,
        passive_accuracy=passive_accuracy,
    )

    plot.savefig(store_location + "/start_set_sizes.pdf")
elif config.wishedPlots == "sampling_strategies":
    latexify(columns=1)
    metrics_list = []
    metrics_list2 = []
    label_list = []
    label_list2 = []
    start_size = 0.01
    batch_size = 150

    labellist = ["Random", "Least Confident", "Entropy", "Margin", "Committee"]
    for i, strategy in enumerate(
        [
            "sheet_random",
            "sheet_uncertainty",
            "sheet_uncertainty_entropy",
            "sheet_uncertainty_max_margin",
            "sheet_committee",
        ]
    ):
        #  for i, strategy in enumerate([
        #  'random', 'uncertainty', 'uncertainty_entropy',
        #  'uncertainty_max_margin', 'committee'
        #  ]):
        pickle_file = (
            config.output
            + "/"
            + strategy
            + "_"
            + str(start_size)
            + "_"
            + str(batch_size)
            + ".pickle"
        )
        print(pickle_file)
        with open(pickle_file, "rb") as f:
            metrics_per_time = pickle.load(f)
        passive_accuracy = get_passive_accuracy(start_size, batch_size)

        test_accs, train_accs = get_metrics(metrics_per_time)

        metrics_list.append(test_accs)
        label_list.append(labellist[i])
        metrics_list2.append(train_accs)
        label_list2.append(labellist[i])

    plot = plot_metrics(
        metrics_list,
        label_list,
        "",
        legend_font_size=7,
        legendLoc=1,
        #  legend=False,
        passive_accuracy=passive_accuracy,
    )

    plot.savefig(store_location + "/sampling_strategies_test.pdf")
    plt.clf()
    plot = plot_metrics(
        metrics_list2,
        label_list2,
        "",
        #  legend=False,
        passive_accuracy=passive_accuracy,
    )

    plot.savefig(store_location + "/sampling_strategies_train.pdf")
elif config.wishedPlots == "batch_sizes":
    latexify(columns=1)
    metrics_list = []
    metrics_list2 = []
    label_list = []
    label_list2 = []
    start_size = 0.01
    strategy = "uncertainty_entropy"

    for batch_size in [50, 150, 250, "query"]:
        label = str(batch_size)
        if batch_size == "query":
            label = "Sheet"
            batch_size = 250
            strategy = "sheet_uncertainty_entropy"
        pickle_file = (
            config.output
            + "/"
            + strategy
            + "_"
            + str(start_size)
            + "_"
            + str(batch_size)
            + ".pickle"
        )
        print(pickle_file)
        with open(pickle_file, "rb") as f:
            metrics_per_time = pickle.load(f)
        passive_accuracy = get_passive_accuracy(start_size, batch_size)

        test_accs, train_accs = get_metrics(metrics_per_time)

        metrics_list.append(test_accs)

        label_list.append(label)
        metrics_list2.append(train_accs)
        label_list2.append(strategy + " (train)")

    plot = plot_metrics(
        metrics_list,
        label_list,
        "",
        #  legend=False,
        xlabel="# Batches",
        passive_accuracy=passive_accuracy,
    )

    plot.savefig(store_location + "/batch_sizes.pdf")

elif config.wishedPlots == "stopping":
    latexify(columns=1)
    metrics_list = []
    label_list = []
    start_size = 0.01
    batch_size = 150
    strategy = config.strategy

    pickle_file = (
        config.output
        + "/"
        + strategy
        + "_"
        + str(start_size)
        + "_"
        + str(batch_size)
        + ".pickle"
    )
    print(pickle_file)
    with open(pickle_file, "rb") as f:
        metrics_per_time = pickle.load(f)
    passive_accuracy = get_passive_accuracy(start_size, batch_size)

    test_accs, train_accs = get_metrics(metrics_per_time)

    metrics_per_time["stop_certainty_list"] = [
        1 - uncertainty for uncertainty in metrics_per_time["stop_certainty_list"]
    ]
    metrics_list.append(test_accs)
    #  metrics_list.append(train_accs)
    metrics_list.append(metrics_per_time["stop_certainty_list"])
    metrics_list.append(metrics_per_time["stop_stddev_list"])
    metrics_list.append(metrics_per_time["stop_accuracy_list"])
    #  label_list.append("LC Uncertainty")
    label_list.append("Committee")
    #  label_list.append(strategy + " (train)")
    label_list.append("Max Uncertainty")
    label_list.append("Standard Deviation")
    label_list.append("Selected Accuracy")
    # calculate based on some criterias somehow the parameters for the stopping criterias
    print("stop")
    pprint(len(metrics_list[0]))

    for measurement in metrics_per_time["stop_certainty_list"]:
        if measurement < config.stop_uncertainty:
            config.stop_uncertainty = metrics_per_time["stop_certainty_list"].index(
                measurement
            )
            break
    for measurement in metrics_per_time["stop_stddev_list"]:
        if measurement < config.stop_std:
            config.stop_std = metrics_per_time["stop_stddev_list"].index(measurement)
            break

    for measurement in metrics_per_time["stop_accuracy_list"]:
        if measurement > config.stop_acc:
            config.stop_acc = metrics_per_time["stop_accuracy_list"].index(measurement)
            break

    # print out stopping criterias

    def slope_calculation(stopping, marker):
        accuracy_start = test_accs[0] * 100
        accuracy_end = test_accs[marker] * 100
        accuracy_delta = accuracy_end - accuracy_start
        nr_queries = marker
        slope = (accuracy_end - accuracy_start) / (nr_queries + 0.0000000000000001)
        print(
            "%s&  %.2f & %.2f & %.2f & %4d & %2.3e \\\\"
            % (
                stopping,
                accuracy_start,
                accuracy_end,
                accuracy_delta,
                nr_queries,
                slope,
            )
        )

    slope_calculation("No Stopping", len(test_accs) - 1)
    print("\\midrule")
    slope_calculation("Maximum Uncertainty Stopping", config.stop_uncertainty)
    slope_calculation("Standard Deviation Stopping", config.stop_std)
    slope_calculation("Selected Accuracy Stopping", config.stop_acc)
    markers_list = [None, config.stop_uncertainty, config.stop_std, config.stop_acc]

    if strategy == "committee":
        xlabel = "#Queried Cell Batches"
        filename = "/stopping_cells.pdf"
    else:
        xlabel = "#Queried Sheets"
        filename = "/stopping.pdf"

    plot = plot_metrics(
        metrics_list,
        label_list,
        "",
        xlabel=xlabel,
        ylabel="Accuracy",
        passive_accuracy=passive_accuracy,
        second_axis="Stopping Criteria",
        markers_list=markers_list,
        highestFirst=True,
    )

    plot.savefig(store_location + filename)
elif config.wishedPlots == "durations":
    latexify(columns=1)

    with open("durations.pickle", "rb") as pickle_file:
        durations = pickle.load(pickle_file)
        durations = [duration / 60000 for duration in sorted(durations)]

        # remove last 5% of elements
        #  print("Removed " + str(int(len(durations) * 0.05)))
        #  pprint(durations[-1])
        #  durations = durations[:-int(len(durations) * 0.05)]

        #  pprint(sorted(durations))
        mu, sigma = norm.fit(durations)
        num_bins = 30
        print("Mean: ", mu, "\t sigma: ", sigma)
        max_duration = np.max(durations)
        print("Max", max_duration)
        median = np.median(durations)
        print("Median:", median)
        n, bins, patches = plt.hist(durations, num_bins)

        # add a 'best fit' line
        #  y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
        #  np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
        #  plt.plot(bins, y, '--')
        plt.xlabel("Duration for classifying a spreadheet in minutes")
        plt.ylabel("Frequency")

        format_axes(plt.gca())
        plt.tight_layout()
        plt.savefig(store_location + "/durations.pdf")
