import abc


class BaseWeakSupervision:
    def __init__(self, data_storage, **THRESHOLDS):
        self.data_storage = data_storage

        for k, v in THRESHOLDS.items():
            setattr(self, k, v)

    @abc.abstractmethod
    def get_labeled_samples(self, **kwargs):
        pass
