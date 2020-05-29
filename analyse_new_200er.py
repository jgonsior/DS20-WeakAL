import pickle
from tabulate import tabulate
from IPython.core.display import display, HTML
import pandas as pd
import seaborn as sns, numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from active_learning.experiment_setup_lib import get_param_distribution

sns.set()
rug = False
bins = 50
# code refactoren und eine funktion drauß machen
# filterung der datentypen refactoren
# so filtern, dass nur die true weaks dabei sind, und davon auch nur die, welche vielversprechende parameterkombinationen enthalten
# bzw. dann auch mal false weaks beibehalten -> es ist keine Erfolgsgarantie!
# ---> Untersuchung, dass ich die Parameter für die Trennung der beiden Bereiche so lange ausprobiere, bis ich den perfekten Wertebereich der Parameter gefunden habe
# -> early Ergebnis an Maik senden


with open("200er_results.pickle", "rb") as f:
    table = pickle.load(f)

df = pd.DataFrame(table)


def compare_two_distributions(
    df, selection1, selection2, label1, label2, axvline=False, display=True, **kwargs,
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
    if not display:
        return
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

    plt.show()


# find distribution, which has the biggest improvement compared to the rest
def calculate_difference(sel1, sel2):
    #  return sel1.median() - sel2.median()
    return sel1.mean() - sel2.mean()


param_dist = get_param_distribution()
highest_diff = 0
highest_sel1 = 0
highest_sel2 = 0

#  zuerst untersuchung auf einzelnen parametern, danach die besten kombinationen kombinieren

#  param = "uncertainty_recommendation_ratio"
#  param = "cluster_recommendation_ratio_labeled_unlabeled"
#  param = "cluster_recommendation_minimum_cluster_unity_size"
#  param = "with_uncertainty_recommendation"
#  param = "with_cluster_recommendation"
param = "uncertainty_recommendation_certainty_threshold"

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
            highest_diff = diff
            highest_sel1 = sel1
            highest_sel2 = sel2


#  compare_two_distributions(
#  df,
#  highest_sel1,
#  highest_sel2,
#  #  df.loc[df["interesting?"] == True]["acc_test"],
#  #  df.loc[df["interesting?"] == False]["acc_test"],
#  "Weak",
#  "No Weak",
#  axvline=True,
#  #  cumulative=True,
#  #  display=False,
#  )
print(df["amount_of_user_asked_queries"])
# display(HTML(tabulate(table, headers="keys", tablefmt="html")))
