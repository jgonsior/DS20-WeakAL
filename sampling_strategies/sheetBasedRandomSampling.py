from pprint import pprint
import random
from itertools import islice
from sampling_strategies.sheetBasedActiveLearner import SheetBasedActiveLearner
import numpy as np


class SheetBasedRandomSampler(SheetBasedActiveLearner):
    def calculate_next_query_indices(self):
        random_sheet_name = random.choice(np.unique(self.X_query_spreadsheets))
        self.current_sheet_name = random_sheet_name
        return np.nonzero(self.X_query_spreadsheets == random_sheet_name)
