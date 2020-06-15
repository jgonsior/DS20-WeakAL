import random

from .baseClusterStrategy import BaseClusterStrategy


class RandomClusterStrategy(BaseClusterStrategy):
    def _get_random_cluster(self):
        # randomly select cluster
        random_cluster = random.choice(
            list(self.data_storage.X_train_unlabeled_cluster_indices.keys())
        )

        #  for cluster, cluster_indices in self.dataset_storage.X_train_unlabeled_cluster_indices.items(
        #  ):
        #  print(cluster, ":\t", len(cluster_indices))
        return random_cluster

    def get_cluster_indices(self, nr_queries_per_iteration, **kwargs):
        random_cluster = self._get_random_cluster()
        #  print("Randomly selected cluster ", random_cluster)
        k = nr_queries_per_iteration
        if k > len(self.data_storage.X_train_unlabeled_cluster_indices[random_cluster]):
            return {
                random_cluster: self.data_storage.X_train_unlabeled_cluster_indices[
                    random_cluster
                ]
            }

        random_indices = random.sample(
            self.data_storage.X_train_unlabeled_cluster_indices[random_cluster], k=k
        )
        return {random_cluster: random_indices}
