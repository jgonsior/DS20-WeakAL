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
        "MostUncertain_entropy" "dummy",
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


#  with open("200er_results.pickle", "rb") as f:
#  with open("1000er_results.pickle", "rb") as f:
with open("old_results.pickle", "rb") as f:
    table = pickle.load(f)

df = pd.DataFrame(table)
df = df.loc[
    #  df.amount_of_user_asked_queries
    #  > 2000
    (df.amount_of_user_asked_queries > 900)
    & (df.amount_of_user_asked_queries < 1100)
]


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


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def find_best_distribution(param, save=False, one_vs_rest_params=False):
    highest_diff = 0
    highest_sel1 = pd.Series([0])
    highest_sel2 = pd.Series([0])
    title = ""

    l = param_dist[param.upper()]

    subsets = []
    if one_vs_rest_params:
        subsets = set(powerset(l))
        subsets.remove(())
        #  subsets.remove(set(l))
    elif df[param].dtypes == bool:
        subsets = [[True]]
    else:
        for lower_bound in l[1:]:
            for upper_bound in l[:-1]:
                if upper_bound <= lower_bound:
                    continue
                subset = set()
                for i in l:
                    if lower_bound <= i and i <= upper_bound:
                        subset.add(i)
                subsets.append(subset)

    for subset in subsets:
        sel1 = sel2 = True
        for element in subset:
            sel1 &= df[param] == element
            sel2 &= df[param] != element
        sel1 = df.loc[sel1]["acc_test"]
        sel2 = df.loc[sel2]["acc_test"]

        #  sel1 = df.loc[(df[param] >= lower_bound) & (df[param] <= upper_bound)][
        #  "acc_test"
        #  ]
        #  sel2 = df.loc[(df[param] < lower_bound) | (df[param] > upper_bound)]["acc_test"]

        diff = calculate_difference(sel1, sel2)
        if diff > highest_diff:
            print(str(subset), "\t\t\t", diff)
            title = param + " kde density plot\nSelection: {} \n Mean Diff: {}".format(
                str(subset), diff
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
            "Param Selection: " + str(len(highest_sel1)),
            "Everything else: " + str(len(highest_sel2)),
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
    #  "interesting?",
    #  "true_weak?",
    #  "acc_test_all_better?",
    "sampling",
]
for param in one_vs_rest_params:
    find_best_distribution(param, True, True)

for param in range_params:
    find_best_distribution(param, True)
