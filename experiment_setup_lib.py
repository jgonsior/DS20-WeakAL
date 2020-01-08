import contextlib
import io
import os
import json
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix


def train_and_evaluate(clf, X_train, Y_train, X_test, Y_test, config):
    training_times = train(clf, X_train, Y_train)
    evaluate(clf,
             X_test,
             Y_test,
             config,
             store=True,
             training_times=training_times)


def train(clf, X_train, Y_train):
    f = io.StringIO()

    with contextlib.redirect_stdout(f):
        clf.fit(X_train, Y_train)

    training_times = f.getvalue()
    return training_times


def evaluate(clf, X_test, Y_test, config, store=False, training_times=""):
    Y_test = Y_test.tolist()

    Y_pred = clf.predict(X_test)
    clf_report = classification_report(Y_test, Y_pred, output_dict=True)
    print(classification_report(Y_test, Y_pred))

    conf_matrix = confusion_matrix(Y_test, Y_pred)
    print(conf_matrix)

    if store:

        # create output folder if not existent
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)

        # save Y_pred
        Y_df = pd.DataFrame(Y_pred)
        Y_df.columns = ['Y_pred']
        Y_df.insert(1, 'Y_test', Y_test)
        Y_df.to_csv(config.output_dir + '/Y_pred.csv', index=None)

        # save classification_report
        with open(config.output_dir + '/results.txt', 'w') as f:
            f.write(json.dumps(clf_report))
            f.write("\n" + "#" * 100 + "\n")
            f.write(json.dumps(conf_matrix.tolist()))
            f.write("\n" + "#" * 100 + "\n")
            f.write(training_times)
