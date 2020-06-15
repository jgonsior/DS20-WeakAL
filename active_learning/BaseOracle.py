from abc import ABC, abstractmethod


class BaseOracle(ABC):
    @abstractmethod
    def get_labeled_samples(self, query_indices, data_storage):
        pass
