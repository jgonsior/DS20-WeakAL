from queue import Queue
from itertools import chain, combinations, permutations
import pickle
from tabulate import tabulate
from IPython.core.display import display, HTML
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit

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
    "INTERESTING?": [True, False],
    "TRUE_WEAK?": [True, False],
}

# code refactoren und eine funktion drauß machen
# filterung der datentypen refactoren
# so filtern, dass nur die true weaks dabei sind, und davon auch nur die, welche vielversprechende parameterkombinationen enthalten
# bzw. dann auch mal false weaks beibehalten -> es ist keine Erfolgsgarantie!
# ---> Untersuchung, dass ich die Parameter für die Trennung der beiden Bereiche so lange ausprobiere, bis ich den perfekten Wertebereich der Parameter gefunden habe
# -> early Ergebnis an Maik senden

#  file = "200er_results.pickle"
#  file = "1000er_results.pickle"
#  file = "2000er_results.pickle"
#  file = "old_results.pickle"
file = "200er_full_results.pickle"

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
        #  df.amount_of_user_asked_queries
        #  == 204
        (df.amount_of_user_asked_queries > 1999)
        #  & (df.amount_of_user_asked_queries < 1100)
    ]


def compare_two_distributions(
    selection_list, axvline=False, save=False, title="", **kwargs,
):
    sns.set(rc={"figure.figsize": (11.7, 8.27)})
    for selection, label in selection_list:
        ax = sns.kdeplot(selection, label=label, **kwargs)

        ax.set_xlim(0.5, 0.875)
        #  ax.set_xlim(0.8, 0.875)
        if axvline:
            ax.axvline(selection.mean(), color=plt.gca().lines[-1].get_color())
    ax.set_title(title)
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
    l = set(param_dist[param.upper()])

    subsets = []
    if one_vs_rest_params:
        for s in l:
            subsets.append([s])
    elif df[param].dtypes == bool or param == "interesting?" or param == "true_weak?":
        subsets.append(([True], [False]))
    else:
        for lower_bound in l:
            for upper_bound in l:
                if upper_bound <= lower_bound:
                    continue
                sel = set()
                for i in l:
                    if lower_bound <= i and i <= upper_bound:
                        sel.add(i)
                if len(sel) == 0 or sel == l:
                    continue
                subsets.append(sel)
    highest_diff = -10000
    highest_sel1 = pd.Series([0])
    highest_sel2 = pd.Series([0])
    highest_s = ""
    title = ""
    selections = []

    sel2 = df.loc[df["true_weak?"] == False]["acc_test"]
    selections.append((sel2, "No Weak: " + str(len(sel2))))
    for s in subsets:
        sel1 = df.loc[df[param].isin(s) & df["true_weak?"] == True]["acc_test"]

        diff = calculate_difference(sel1, sel2)

        if one_vs_rest_params:
            selections.append((sel1, str(s) + ": " + str(len(sel1))))

        if diff > highest_diff:
            print(str(s), "\t\t\t", diff)
            title = " kde density plot\nSelection: {} \n Mean Diff: {}".format(
                str(s), diff
            )
            highest_diff = diff
            highest_sel1 = sel1
            highest_sel2 = sel2
            if type(next(iter(s))) == np.float64:
                s = "{}-{}".format(min(s), max(s))
                s2 = "Rest"
            highest_s = str(s)

    if not one_vs_rest_params:
        selections.append((highest_sel1, highest_s + ": " + str(len(highest_sel1))))

    if save:
        compare_two_distributions(
            selections,
            axvline=True,
            title="{:.2%}".format(highest_diff) + "\n" + param + "\n" + highest_s,
            save="True",
        )
    return highest_diff, highest_sel1, highest_sel2, title


# ich habe jetzt DIE eine insgesamt beste Parameterkombination -> der nächste Schritt sind ranges, und danach subsets um mehrere Mengen von guten Kombis zu finden
def recursive_hyper_search(param_list, sel, baseline, df, sel_dict):
    if len(param_list) == 0 or len(df.loc[sel]) == 0:
        selection = df.loc[sel]["acc_test"]
        #  print(selection)
        #  print(baseline)
        score = calculate_difference(selection, baseline)
        #  print(sel_dict, score)
        return score, sel_dict, len(selection)
    max_score = last_score = np.float("-inf")
    max_sel = None
    max_len = None
    lower_bound_reached = upper_bound_reached = False

    original_params = param_dist[param_list[0].upper()]
    q = Queue()

    if isinstance(original_params[0], np.float64):
        q.put((min(original_params), max(original_params)))
    else:
        for subset in powerset(original_params):
            q.put(set(subset))
    while not q.empty():
        value = q.get(block=False)
        #  print("{}: {}".format(param_list[0], value))
        if type(value) == set:
            sel_new = sel & (df[param_list[0]].isin(value))
        else:
            lower_bound, upper_bound = value
            sel_new = (
                sel
                & (df[param_list[0]] >= lower_bound)
                & (df[param_list[0]] <= upper_bound)
            )

        sel_dict[param_list[0]] = value
        score, best_sel, length = recursive_hyper_search(
            param_list[1:], sel_new, baseline, df, sel_dict
        )
        #  print("{} {}".format(last_score, score))
        if score > max_score:
            max_score = score
            max_sel = sel_dict.copy()
            max_len = length
        if not upper_bound_reached:
            if score + 0.01 >= last_score and type(value) != set:
                q.put((lower_bound + 0.01, upper_bound))
            else:
                upper_bound_reached = True
        if not lower_bound_reached:
            if score + 0.01 >= last_score and type(value) != set:
                q.put((lower_bound, upper_bound - 0.01))
            else:
                lower_bound_reached = True

        last_score = score

    return max_score, max_sel, max_len


def find_multiple_hyper_param_combinations(params):
    baseline = df.loc[df["true_weak?"] == False]["acc_test"]

    sel = df["true_weak?"] == True
    print(recursive_hyper_search(params, sel, baseline, df, {}))


def get_distributions_for_interesting(params):
    baseline = df.loc[df["true_weak?"] == False]["acc_test"]
    true_interesting = df.loc[df["interesting?"] == True]["acc_test"]
    false_interesting = df.loc[df["interesting?"] == False]["acc_test"]
    highest_diff = calculate_difference(true_interesting, baseline)
    compare_two_distributions(
        [
            (baseline, "No Weak: {:>4} {:.2%}".format(len(baseline), baseline.mean())),
            (
                true_interesting,
                "Weak and improvement:{:>4} {:.2%}".format(
                    len(true_interesting), true_interesting.mean()
                ),
            ),
            (
                false_interesting,
                "Weak and no improvement:{:>4} {:.2%}".format(
                    len(false_interesting), false_interesting.mean()
                ),
            ),
        ],
        axvline=True,
        title='Difference mean "Weak and improvement" to "No Weak ":{:.2%}'.format(
            highest_diff
        ),
        save=True,
    )

    true_interesting = df.loc[df["interesting?"] == True]

    #  für alpha, beta, gamma jointplots über ganzen Wertebereich, mit acc_test als highlight farbe?
    cmap = sns.cubehelix_palette(start=0.0, rot=-0.75, as_cmap=True)
    #  cmap = sns.color_palette("cubehelix")
    true_interesting["acc_test"] = true_interesting["acc_test"].multiply(100)
    for a, b in permutations(params[0], 2):
        sns.scatterplot(
            x=a,
            y=b,
            data=true_interesting,
            palette=cmap,
            #  sizes=[45, 60, 75, 90],
            hue="acc_test",
            size="acc_test",
        )
        plt.tight_layout()
        plt.savefig("plots/{}_{}".format(a, b))
        #  plt.show()
        plt.clf()

    for param in params[0]:
        selections = []

        # für die balken mehrerer zusammen nehmen
        for value in param_dist[param.upper()]:
            sel = true_interesting.loc[true_interesting[param] == value]["acc_test"]
            selections.append(
                (sel, "{:>5} : {:>6} - {:.2%}".format(value, len(sel), sel.mean()))
            )
        compare_two_distributions(
            selections, axvline=True, title=param, save=True,
        )


range_params = [
    "uncertainty_recommendation_ratio",
    "cluster_recommendation_ratio_labeled_unlabeled",
    "cluster_recommendation_minimum_cluster_unity_size",
    "uncertainty_recommendation_certainty_threshold",
]

one_vs_rest_params = [
    "cluster",
    "sampling",
    "interesting?",
    "true_weak?",
    "with_uncertainty_recommendation",
    "with_cluster_recommendation",
]


hyper_test_params = [
    [
        "sampling",
        "cluster",
        "cluster_recommendation_ratio_labeled_unlabeled",
        "uncertainty_recommendation_certainty_threshold",
        "cluster_recommendation_minimum_cluster_unity_size",
        #  "uncertainty_recommendation_ratio",
    ],
    #  [
    #      "sampling",
    #      "cluster",
    #      "uncertainty_recommendation_ratio",
    #      "cluster_recommendation_ratio_labeled_unlabeled",
    #      "cluster_recommendation_minimum_cluster_unity_size",
    #      "uncertainty_recommendation_certainty_threshold",
    #  ],
]

get_distributions_for_interesting(hyper_test_params)


#  for params in hyper_test_params:
#      find_multiple_hyper_param_combinations(params)

#  for param in one_vs_rest_params:
#      find_best_distribution(param, True, True)
#
#  for param in range_params:
#      find_best_distribution(param, True)
