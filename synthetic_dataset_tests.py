import csv
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from experiment_setup_lib import get_dataset

plt.style.use("seaborn")


def test_run(random_run):
    N_SAMPLES = 1000
    N_FEATURES = random.randint(10, 100)
    N_INFORMATIVE, N_REDUNDANT, N_REPEATED = [
        int(N_FEATURES * i) for i in np.random.dirichlet(np.ones(3), size=1).tolist()[0]
    ]

    N_CLASSES = random.randint(2, 10)
    N_CLUSTERS_PER_CLASS = random.randint(
        1, min(max(1, int(2 ** N_INFORMATIVE / N_CLASSES)), 10)
    )

    if N_CLASSES * N_CLUSTERS_PER_CLASS > 2 ** N_INFORMATIVE:
        return

    WEIGHTS = np.random.dirichlet(np.ones(N_CLASSES), size=1).tolist()[
        0
    ]  # list of weights, len(WEIGHTS) = N_CLASSES, sum(WEIGHTS)=1
    FLIP_Y = 0.01  # amount of noise, larger values make it harder
    CLASS_SEP = random.uniform(
        0, 10
    )  # larger values spread out the clusters and make it easier
    HYPERCUBE = True  # if false random polytope
    SCALE = 0.01  # features should be between 0 and 1 now

    synthetic_creation_args = {
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
        "n_informative": N_INFORMATIVE,
        "n_redundant": N_REDUNDANT,
        "n_repeated": N_REPEATED,
        "n_classes": N_CLASSES,
        "n_clusters_per_class": N_CLUSTERS_PER_CLASS,
        "weights": WEIGHTS,
        "flip_y": FLIP_Y,
        "class_sep": CLASS_SEP,
        "hypercube": HYPERCUBE,
        "scale": SCALE,
    }
    print(random_run, ": ", synthetic_creation_args)

    X_train, X_test, Y_train, Y_test, label_encoder_classes = get_dataset(
        "../datasets", "synthetic", **synthetic_creation_args
    )

    BATCH_SIZE = 10
    accs = []
    clf = RandomForestClassifier()

    for i in range(1, int(len(Y_train) / BATCH_SIZE)):
        train_data_limit = i * BATCH_SIZE
        clf.fit(
            X_train[:train_data_limit].to_numpy(),
            Y_train[:train_data_limit][0].to_numpy(),
            sample_weight=compute_sample_weight(
                "balanced", Y_train[:train_data_limit][0].to_numpy()
            ),
        )

        Y_pred = clf.predict(X_test.to_numpy().astype(np.float64))

        current_acc = accuracy_score(Y_test, Y_pred)
        #  print("{} {}".format(train_data_limit, current_acc))
        accs.append(current_acc)

    synthetic_creation_args["accs"] = accs
    with open("plots/results.csv", "a") as f:
        w = csv.DictWriter(f, synthetic_creation_args.keys())
        w.writerow(synthetic_creation_args)
    del synthetic_creation_args["accs"]

    markers_on = [np.argmax(accs)]

    synthetic_creation_args["weights"] = [
        round(i, 2) for i in synthetic_creation_args["weights"]
    ]
    synthetic_creation_args["class_sep"] = round(
        synthetic_creation_args["class_sep"], 2
    )

    plt.plot(
        [i * BATCH_SIZE for i in range(1, int(len(Y_train) / BATCH_SIZE))],
        accs,
        "-gD",
        markevery=markers_on,
    )

    plt.ylim(0, 1)

    plt.legend(
        ["\n".join([k + ": " + str(v) for k, v in synthetic_creation_args.items()])]
    )
    #  plt.show()
    plt.savefig("plots/" + str(random_run) + ".png")
    plt.clf()


#  with parallel_backend("loky", n_jobs=1):
Parallel(n_jobs=8)(delayed(test_run)(i) for i in range(0, 100))
