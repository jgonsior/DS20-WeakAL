import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_sample_weight

from ..activeLearner import ActiveLearner


class Committee:
    def __init__(self, clf_list):
        self._clf_list = clf_list

    @property
    def clf_list(self):
        return self._clf_list

    def fit(self, X_train, Y_train):
        for _, clf in enumerate(self.clf_list):
            clf.fit(
                X_train,
                Y_train,
                sample_weight=compute_sample_weight("balanced", Y_train),
            )

    def predict(self, X_test):
        committee_predictions = []
        for clf in self.clf_list:
            committee_predictions.append(clf.predict(X_test))
        return committee_predictions

    def get_clf_list(self):
        return self._clf_list


class CommitteeSampler(ActiveLearner):
    def __init__(self, config):

        super(CommitteeSampler, self).__init__(config)

        self.best_hyper_parameters = {"n_jobs": config.cores}

        random_state = config.random_seed

        clf0 = RandomForestClassifier(
            random_state=37264 * random_state, **self.best_hyper_parameters
        )
        clf1 = RandomForestClassifier(
            random_state=948 * random_state, **self.best_hyper_parameters
        )
        clf2 = MultinomialNB()
        clf3 = svm.SVC(random_state=2648 * random_state, gamma="auto")
        clf4 = RandomForestClassifier(
            random_state=382 * random_state, **self.best_hyper_parameters
        )

        self.clf_list = [clf0, clf1, clf2, clf3, clf4]

        self.committee = Committee(self.clf_list)

        # copy and paste because lazy
        # with new clf_list!
        self.metrics_per_al_cycle = {
            "test_data_metrics": [[] for clf in self.clf_list],
            "train_labeled_data_metrics": [[] for clf in self.clf_list],
            "train_unlabeled_data_metrics": [[] for clf in self.clf_list],
            "train_unlabeled_class_distribution": [[] for clf in self.clf_list],
            "stop_certainty_list": [],
            "stop_stddev_list": [],
            "stop_accuracy_list": [],
            "query_length": [],
        }

    def fit_clf(self,):
        self.committee.fit(self.X_train_labeled, self.Y_train_labeled)

    def setClassifierClasses(self, classes):
        self.classifier_classes = classes

    def calculate_next_query_indices(self):

        committee_predictions = self.committee.predict(self.X_train_unlabeled)

        len_predictions = len(committee_predictions[0])
        count_clf = len(self.clf_list)
        score = np.zeros(len_predictions)

        for i in range(len_predictions):
            class_count = [0] * len(self.classifier_classes)
            for clf_i in range(count_clf):
                ordinal = self.classifier_classes.index(committee_predictions[clf_i][i])
                class_count[ordinal] += 1
            score[i] = sum([c * c for c in class_count])

        return score.argsort()[: self.nr_queries_per_iteration]
