import numpy as np
from scipy.stats import entropy

from .sheetBasedActiveLearner import SheetBasedActiveLearner


class SheetBasedUncertaintySampler(SheetBasedActiveLearner):
    def set_uncertainty_strategy(self, strategy):
        self.strategy = strategy

    def setClassifierClasses(self, classes):
        self.classifier_classes = classes

    def calculate_next_query_indices(self):
        X_query = self.X_train_unlabeled

        # recieve predictions and probabilities
        # for all possible classifications of CLASSIFIER
        Y_temp_proba = self.clf_list[0].predict_proba(X_query)

        if self.strategy == "least_confident":
            result = 1 - np.amax(Y_temp_proba, axis=1)
        elif self.strategy == "max_margin":
            margin = np.partition(-Y_temp_proba, 1, axis=1)
            result = -np.abs(margin[:, 0] - margin[:, 1])
        elif self.strategy == "entropy":
            result = np.apply_along_axis(entropy, 1, Y_temp_proba)

        class_proba = -result

        # group by spreadsheets
        # generate {spreadsheet => [class_proba]}
        # after that generate mean of class_proba per spreadsheet and then take all indices from self.X_query_spreadsheets as result

        spreadsheet_to_class_proba = {}

        for spreadsheet, class_probability in zip(
            self.X_query_spreadsheets, class_proba
        ):
            if spreadsheet not in spreadsheet_to_class_proba.keys():
                spreadsheet_to_class_proba[spreadsheet] = []
            spreadsheet_to_class_proba[spreadsheet].append(class_probability)

        for spreadsheet, class_probabilities in spreadsheet_to_class_proba.items():
            spreadsheet_to_class_proba[spreadsheet] = np.mean(class_probabilities)

        most_uncertain_spreadsheet = min(
            spreadsheet_to_class_proba, key=spreadsheet_to_class_proba.get
        )

        self.current_sheet_name = most_uncertain_spreadsheet

        return np.nonzero(self.X_query_spreadsheets == most_uncertain_spreadsheet)
