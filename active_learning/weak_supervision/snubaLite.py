import itertools
import collections
import random
import numpy as np

from ..activeLearner import ActiveLearner
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from .baseWeakSupervision import BaseWeakSupervision


class SnubaLite(BaseWeakSupervision):
    def get_labeled_samples(self):
        X_weak = Y_weak = weak_indices = None
        # @todo prevent snuba_lite from relearning based on itself (so only "strong" labels are being used for weak labeling)
        # for each label and each feature (or feature combination) generate small shallow decision tree -> is it a good idea to limit the amount of used features?!
        highest_accuracy = 0
        best_heuristic = None
        best_combination = None
        best_class = None

        combinations = []
        for combination in itertools.combinations(
            list(range(self.data_storage.X_train_labeled.shape[1])), 1
        ):
            combinations.append(combination)

        # generated heuristics should only being applied to small subset (which one?)
        # balance jaccard and f1_measure (coverage + accuracy)
        for clf_class in self.data_storage.label_encoder.classes_:
            for combination in combinations:
                # create training and test data set out of current available training/test data
                X_temp = self.data_storage.X_train_labeled.loc[:, combination]

                # do one vs rest
                Y_temp = self.data_storage.Y_train_labeled.copy()
                Y_temp = Y_temp.replace(
                    self.data_storage.label_encoder.transform(
                        [
                            c
                            for c in self.data_storage.label_encoder.classes_
                            if c != clf_class
                        ]
                    ),
                    -1,
                )

                X_temp_train, X_temp_test, Y_temp_train, Y_temp_test = train_test_split(
                    X_temp, Y_temp, train_size=0.6
                )

                heuristic = DecisionTreeClassifier(max_depth=2)
                heuristic.fit(X_temp_train, Y_temp_train)

                accuracy = accuracy_score(Y_temp_test, heuristic.predict(X_temp_test))
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    best_heuristic = heuristic
                    best_combination = combination
                    best_class = clf_class

        # if accuracy of decision tree is high enough -> take recommendation
        if highest_accuracy > self.MINIMUM_HEURISTIC_ACCURACY:
            probabilities = best_heuristic.predict_proba(
                self.data_storage.X_train_unlabeled.loc[:, best_combination].to_numpy()
            )

            # filter out labels where one-vs-rest heuristic is sure that sample is of label L
            weak_indices = [
                index
                for index, proba in zip(
                    self.data_storage.X_train_unlabeled.index, probabilities
                )
                if np.max(proba) > self.MINIMUM_HEURISTIC_ACCURACY
            ]

            if len(weak_indices) > 0:
                #  log_it("Snuba mit Klasse " + best_class)
                #  log_it(weak_indices)
                X_weak = self.data_storage.X_train_unlabeled.loc[weak_indices]
                best_class_encoded = self.data_storage.label_encoder.transform(
                    [best_class]
                )[0]
                Y_weak = [best_class_encoded for _ in weak_indices]
            else:
                weak_indices = None

        return X_weak, Y_weak, weak_indices, "S"
