import random

import numpy as np

from .sheetBasedActiveLearner import SheetBasedActiveLearner


class SheetBasedRandomSampler(SheetBasedActiveLearner):
    def calculate_next_query_indices(self):
        random_sheet_name = random.choice(np.unique(self.X_query_spreadsheets))
        self.current_sheet_name = random_sheet_name
        return np.nonzero(self.X_query_spreadsheets == random_sheet_name)
