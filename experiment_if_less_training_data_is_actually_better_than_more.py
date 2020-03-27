import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from joblib import Parallel, delayed
from experiment_setup_lib import get_dataset
from joblib import parallel_backend

np.set_printoptions(threshold=sys.maxsize)

X_train, X_test, Y_train, Y_test, label_encoder_classes = get_dataset(
    "../datasets", "dwtc"
)
clf = RandomForestClassifier()

clf.fit(
    X_train.to_numpy(),
    Y_train[0].to_numpy(),
    sample_weight=compute_sample_weight("balanced", Y_train[0].to_numpy()),
)

Y_pred = clf.predict(X_test)
current_acc = accuracy_score(Y_test, Y_pred)
print("Passive: ", current_acc)
print(label_encoder_classes)
del clf


def test_run(random_run):
    BATCH_SIZE = 100
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
        print("{} {}".format(train_data_limit, current_acc))
        accs.append(current_acc)
    plt.style.use("seaborn")

    markers_on = [np.argmax(accs)]

    plt.plot(
        [i * BATCH_SIZE for i in range(1, int(len(Y_train) / BATCH_SIZE))],
        accs,
        "-gD",
        markevery=markers_on,
    )
    plt.ylim(0, 1)
    plt.savefig("plots/" + str(random_run) + ".png")
    plt.clf()


with parallel_backend("loky", n_jobs=8):
    Parallel()(delayed(test_run)(i) for i in range(0, 500))
