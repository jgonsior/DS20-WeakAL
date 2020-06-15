from abc import ABC, abstractmethod


class BaseOracle(ABC):
    @abstractmethod
    def get_labels(self, query_indices, data_storage):
        pass
