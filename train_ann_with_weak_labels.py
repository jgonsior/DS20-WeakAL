import os
import numpy as np
from active_learning.al_cycle_wrapper import train_and_eval_dataset
from active_learning.experiment_setup_lib import get_dataset, standard_config
from fake_experiment_oracle import FakeExperimentOracle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from tensorflow import keras

config = standard_config([
    (["--DATASET_NAME"], {
        "required": True,
    }),
    (["--PICKLE"], {
        "required": True,
    }),
])

# read in data the same as before (same random inidce)
X_train, X_test, Y_train, Y_test, label_encoder_classes = get_dataset(
    config.DATASETS_PATH, config.DATASET_NAME, config.RANDOM_SEED)

for filename in os.listdir(config.PICKLE):
    Y_train_al = pd.read_pickle(config.PICKLE + "/" + filename)
    # calculate accuracy_score between Y_train_al and Y_train_real
    amount_of_labels = len(Y_train_al)
    accuracy = accuracy_score(Y_train.iloc[Y_train_al.index], Y_train_al[0])
    percentage_user_asked_queries = amount_of_labels / 2888

    combined_score = (2 * percentage_user_asked_queries * accuracy /
                      (percentage_user_asked_queries + accuracy))

    if combined_score > 0.4:
        #  if amount_of_labels > 1000 and accuracy > 0.8:
        # calculate acc per source
        #  for source in Y_train_al.source.unique():
        #  ys_source = Y_train_al.loc[Y_train_al.source == source]
        #  print(source + " Len: {:>4} Acc: {:.2f}".format(
        #  len(ys_source),
        #  accuracy_score(
        #  Y_train.iloc[ys_source.index],
        #  ys_source[0],
        #  ),
        #  ))

        # calculate false baseline, result of random forest on all
        weak_rf = RandomForestClassifier(random_state=config.RANDOM_SEED,
                                         n_jobs=20)
        weak_rf.fit(X_train.iloc[Y_train_al.index], Y_train_al[0])
        weak_acc = accuracy_score(Y_train, weak_rf.predict(X_train))

        # calculate false baseline, result of random forest only on active
        active_rf = RandomForestClassifier(random_state=config.RANDOM_SEED,
                                           n_jobs=20)
        ys_oracle = Y_train_al.loc[Y_train_al.source == "A"]
        active_rf.fit(X_train.iloc[ys_oracle.index], ys_oracle[0])
        orac_acc = accuracy_score(Y_train, active_rf.predict(X_train))

        if weak_acc > orac_acc or weak_acc > 0.44:
            print(filename)
            print("Combined score {:.2f}".format(combined_score))
            print("X Len: {:>4} Acc: {:.2f}".format(amount_of_labels,
                                                    accuracy))
            print("Orac RF {:.2f}".format(orac_acc))
            print("Weak RF {:.2f}".format(weak_acc))
            print("\n")

            # train ann, first with active batches, than with weak batches?
            model = keras.Sequential([
                keras.layers.Dense(64,
                                   activation="relu",
                                   input_shape=(X_train.shape[-1], )),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(1, activation="sigmoid"),
            ])

            print(model.summary())

            metrics = [
                keras.metrics.Accuracy(name="accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ]

            model.compile(optimizer=keras.optimizers.Adam(1e-2))

            model.fit(
                X_train.iloc[Y_train_al.index],
                Y_train_al[0],
                batch_size=128,
                epochs=50,
                verbose=3,
                validation_data=(X_test, Y_test),
            )
