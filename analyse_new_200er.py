import pickle
from tabulate import tabulate
from IPython.core.display import display, HTML
import pandas as pd
import seaborn as sns, numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from active_learning.experiment_setup_lib import get_param_distribution

param_dist = get_param_distribution()
# code refactoren und eine funktion drauß machen
# filterung der datentypen refactoren
# so filtern, dass nur die true weaks dabei sind, und davon auch nur die, welche vielversprechende parameterkombinationen enthalten
# bzw. dann auch mal false weaks beibehalten -> es ist keine Erfolgsgarantie!
# ---> Untersuchung, dass ich die Parameter für die Trennung der beiden Bereiche so lange ausprobiere, bis ich den perfekten Wertebereich der Parameter gefunden habe
# -> early Ergebnis an Maik senden


#  with open("1000er_results.pickle", "rb") as f:
with open("200er_results.pickle", "rb") as f:
    table = pickle.load(f)

df = pd.DataFrame(table)


def compare_two_distributions(
    df,
    selection1,
    selection2,
    label1,
    label2,
    axvline=False,
    save=False,
    title="",
    **kwargs,
):
    print("\t #Label  Mean \t\t\t Median")
    print(
        label1 + ": \t",
        len(selection1),
        "\t",
        selection1.mean(),
        "\t",
        selection1.median(),
    )
    print(
        label2 + ":",
        len(selection2),
        "\t",
        selection2.mean(),
        "\t",
        selection2.median(),
    )
    print(
        "Differenz\t",
        selection1.mean() - selection2.mean(),
        "\t",
        selection1.median() - selection2.median(),
    )
    ax1 = sns.kdeplot(selection1, label=label1, color="orange", **kwargs)

    ax1.set_xlim(0.5, 0.9)
    if axvline:
        ax1.axvline(selection1.mean(), color="orange")
    # plt.show()
    # plt.clf()

    ax2 = sns.kdeplot(selection2, label=label2, color="blue", **kwargs)
    ax2.set_xlim(0.5, 0.9)

    if axvline:
        ax2.axvline(selection2.mean(), color="blue")
    ax2.set_title(title)
    plt.tight_layout()
    if save:
        plt.savefig("plots/" + title.replace("\n", "_").replace(" ", "") + ".pdf")
        plt.savefig("plots/" + title.replace("\n", "_").replace(" ", "") + ".png")
    else:
        plt.show()
    plt.clf()


# find distribution, which has the biggest improvement compared to the rest
def calculate_difference(sel1, sel2):
    #  return sel1.median() - sel2.median()
    return sel1.mean() - sel2.mean()


def find_best_distribution(param, save=False):
    highest_diff = 0
    highest_sel1 = 0
    highest_sel2 = 0
    title = ""

    l = param_dist[param.upper()]
    for index, lower_bound in enumerate(l):
        for index, upper_bound in enumerate(l):

            if df[param].dtypes == bool:
                sel1 = df.loc[df[param] == lower_bound]["acc_test"]
                sel2 = df.loc[df[param] != lower_bound]["acc_test"]
            else:
                if upper_bound <= lower_bound:
                    continue
                sel1 = df.loc[(df[param] >= lower_bound) & (df[param] <= upper_bound)][
                    "acc_test"
                ]
                sel2 = df.loc[(df[param] < lower_bound) | (df[param] > upper_bound)][
                    "acc_test"
                ]

            diff = calculate_difference(sel1, sel2)
            if diff > highest_diff:
                print(lower_bound, upper_bound, "\t\t\t", diff)
                title = (
                    param
                    + " kde density plot\nSelection: {} - {} \n Mean Diff: {}".format(
                        lower_bound, upper_bound, diff
                    )
                )
                highest_diff = diff
                highest_sel1 = sel1
                highest_sel2 = sel2

    if save:
        compare_two_distributions(
            df,
            highest_sel1,
            highest_sel2,
            #  df.loc[df["interesting?"] == True]["acc_test"],
            #  df.loc[df["interesting?"] == False]["acc_test"],
            "Param Selection",
            "Everything else",
            axvline=True,
            title=title,
            save="True",
        )
    return highest_diff, highest_sel1, highest_sel2, title


#  param = "uncertainty_recommendation_ratio"
#  param = "cluster_recommendation_ratio_labeled_unlabeled"
#  param = "cluster_recommendation_minimum_cluster_unity_size"
#  param = "with_uncertainty_recommendation"
#  param = "with_cluster_recommendation"
#  param = "uncertainty_recommendation_certainty_threshold"
params = [
    "uncertainty_recommendation_ratio",
    "cluster_recommendation_ratio_labeled_unlabeled",
    "cluster_recommendation_minimum_cluster_unity_size",
    "with_uncertainty_recommendation",
    "with_cluster_recommendation",
    "uncertainty_recommendation_certainty_threshold",
    #  "interesting?",
    #  "true_weak?",
    #  "acc_test_all_better?",
]
for param in params:
    find_best_distribution(param, True)
