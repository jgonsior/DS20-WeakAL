import numpy as np
from pprint import pprint
from activeLearner import ActiveLearner
from scipy.stats import entropy


class UncertaintySampler(ActiveLearner):
    def set_uncertainty_strategy(self, strategy):
        self.strategy = strategy

    def setClassifierClasses(self, classes):
        self.classifier_classes = classes

    def calculate_next_query_indices(self):
        # recieve predictions and probabilitys
        # for all possible classifications of classifier
        Y_temp_pred = self.clf_list[0].predict(self.X_train_unlabeled)
        Y_temp_proba = self.clf_list[0].predict_proba(self.X_train_unlabeled)

        if self.strategy == 'least_confident':
            result = 1 - np.amax(Y_temp_proba, axis=1)
        elif self.strategy == 'max_margin':
            margin = np.partition(-Y_temp_proba, 1, axis=1)
            result = -np.abs(margin[:, 0] - margin[:, 1])
        elif self.strategy == 'entropy':
            result = np.apply_along_axis(entropy, 1, Y_temp_proba)

        #  pprint(result)
        print(np.argsort(-result))

        query_indices = np.argsort(-result)[:self.nr_queries_per_iteration]
        # return smallest probabilities
        return query_indices
