from itertools import chain, combinations
import pickle
from tabulate import tabulate
from IPython.core.display import display, HTML
import pandas as pd
import seaborn as sns, numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

param_size = 50
#  param_size = 2
zero_to_one = np.linspace(0, 1, num=param_size * 2 + 1).astype(float)
half_to_one = np.linspace(0.5, 1, num=param_size + 1).astype(float)


param_dist = {
    #  "DATASETS_PATH": [DATASETS_PATH],
    #  "CLASSIFIER": [CLASSIFIER],
    #  "N_JOBS": [N_JOBS],
    #  "RANDOM_SEED": [RANDOM_SEED],
    #  "TEST_FRACTION": [TEST_FRACTION],
    "SAMPLING": [
        "random",
        "uncertainty_lc",
        "uncertainty_max_margin",
        "uncertainty_entropy",
    ],
    "CLUSTER": [
        "dummy",
        "random",
        "MostUncertain_lc",
        "MostUncertain_max_margin",
        "MostUncertain_entropy",
    ],
    #  "NR_LEARNING_ITERATIONS": [NR_LEARNING_ITERATIONS],
    #  "NR_LEARNING_ITERATIONS": [1],
    #  "NR_QUERIES_PER_ITERATION":
    #  NR_QUERIES_PER_ITERATION,
    #  "START_SET_SIZE":
    #  START_SET_SIZE,
    "STOPPING_CRITERIA_UNCERTAINTY": [1],  # zero_to_one,
    "STOPPING_CRITERIA_STD": [1],  # zero_to_one,
    "STOPPING_CRITERIA_ACC": [1],  # zero_to_one,
    "ALLOW_RECOMMENDATIONS_AFTER_STOP": [True],
    # uncertainty_recommendation_grid = {
    "UNCERTAINTY_RECOMMENDATION_CERTAINTY_THRESHOLD": half_to_one,
    "UNCERTAINTY_RECOMMENDATION_RATIO": [
        1 / 100,
        1 / 1000,
        1 / 10000,
        1 / 100000,
        1 / 1000000,
    ],
    # snuba_lite_grid = {
    "SNUBA_LITE_MINIMUM_HEURISTIC_ACCURACY": [0],
    #  half_to_one,
    # cluster_recommendation_grid = {
    "CLUSTER_RECOMMENDATION_MINIMUM_CLUSTER_UNITY_SIZE": half_to_one,
    "CLUSTER_RECOMMENDATION_RATIO_LABELED_UNLABELED": half_to_one,
    "WITH_UNCERTAINTY_RECOMMENDATION": [True, False],
    "WITH_CLUSTER_RECOMMENDATION": [True, False],
    "WITH_SNUBA_LITE": [False],
    "MINIMUM_TEST_ACCURACY_BEFORE_RECOMMENDATIONS": half_to_one,
    #  "DB_NAME_OR_TYPE": [DB_NAME_OR_TYPE],
    "USER_QUERY_BUDGET_LIMIT": [2000],
}

# code refactoren und eine funktion drauß machen
# filterung der datentypen refactoren
# so filtern, dass nur die true weaks dabei sind, und davon auch nur die, welche vielversprechende parameterkombinationen enthalten
# bzw. dann auch mal false weaks beibehalten -> es ist keine Erfolgsgarantie!
# ---> Untersuchung, dass ich die Parameter für die Trennung der beiden Bereiche so lange ausprobiere, bis ich den perfekten Wertebereich der Parameter gefunden habe
# -> early Ergebnis an Maik senden

file = "200er_results.pickle"
file = "1000er_results.pickle"
file = "2000er_results.pickle"
#  file = "old_results.pickle"

if file != "old_results.pickle":
    param_dist["UNCERTAINTY_RECOMMENDATION_CERTAINTY_THRESHOLD"] = np.linspace(
        0.85, 1, num=15 + 1
    )

with open(file, "rb") as f:
    #  with open("1000er_results.pickle", "rb") as f:
    #  with open("old_results.pickle", "rb") as f:
    table = pickle.load(f)

df = pd.DataFrame(table)

if file == "old_results.pickle":
    df = df.loc[
        df.amount_of_user_asked_queries
        == 204
        #  (df.amount_of_user_asked_queries > 900)
        #  & (df.amount_of_user_asked_queries < 1100)
    ]


def compare_two_distributions(
    non_weak,
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

    ax3 = sns.kdeplot(non_weak, label="Non weak", color="grey", **kwargs)
    ax3.set_xlim(0.5, 0.9)

    if axvline:
        ax3.axvline(non_weak.mean(), color="grey")

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


def powerset(s):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    #  s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def find_best_distribution(param, save=False, one_vs_rest_params=False):
    highest_diff = -10000
    highest_sel1 = pd.Series([0])
    highest_sel2 = pd.Series([0])
    highest_s1 = highest_s2 = ""
    title = ""

    l = set(param_dist[param.upper()])

    subsets = []
    if one_vs_rest_params:
        for s in set(powerset(l)):
            if s == ():
                continue
            s = set(s)
            subsets.append((s, l.difference(s)))
        #  subsets.remove(set(l))
    elif df[param].dtypes == bool:
        subsets.append(([True], [False]))
    else:
        for lower_bound in l:
            for upper_bound in l:
                if upper_bound <= lower_bound:
                    continue
                sel1 = set()
                sel2 = set()
                for i in l:
                    if lower_bound <= i and i <= upper_bound:
                        sel1.add(i)
                    else:
                        sel2.add(i)
                if len(sel1) == 0 or len(sel2) == 0:
                    continue
                subsets.append((sel1, sel2))

    for s1, s2 in subsets:
        sel1 = df.loc[df[param].isin(s1) & df["true_weak?"] == True]["acc_test"]
        sel2 = df.loc[~(df[param].isin(s1) & df["true_weak?"] == True)]["acc_test"]
        diff = calculate_difference(sel1, sel2)
        if diff > highest_diff:
            print(str(s1), "\t\t\t", diff)
            title = " kde density plot\nSelection: {} \n Mean Diff: {}".format(
                str(s1), diff
            )
            highest_diff = diff
            highest_sel1 = sel1
            highest_sel2 = sel2
            if type(next(iter(s1))) == np.float64:
                s1 = "{}-{}".format(min(s1), max(s1))
                s2 = "Rest"
            highest_s1 = str(s1)
            highest_s2 = str(s2)

    if save:
        compare_two_distributions(
            df.loc[df["true_weak?"] == True]["acc_test"],
            highest_sel1,
            highest_sel2,
            highest_s1 + ": " + str(len(highest_sel1)),
            highest_s2 + ": " + str(len(highest_sel2)),
            axvline=True,
            title="{}".format(highest_diff) + param,
            save="True",
        )
    return highest_diff, highest_sel1, highest_sel2, title


range_params = [
    "uncertainty_recommendation_ratio",
    "cluster_recommendation_ratio_labeled_unlabeled",
    "cluster_recommendation_minimum_cluster_unity_size",
    "with_uncertainty_recommendation",
    "with_cluster_recommendation",
    "uncertainty_recommendation_certainty_threshold",
]

one_vs_rest_params = [
    "cluster",
    "sampling",
]
for param in one_vs_rest_params:
    find_best_distribution(param, True, True)

for param in range_params:
    find_best_distribution(param, True)
