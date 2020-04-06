from active_learning.BaseOracle import BaseOracle


class FakeExperimentOracle(BaseOracle):
    def get_labels(self, query_indices, data_storage):
        return data_storage.Y_train_unlabeled.loc[query_indices]
