from pprint import pprint

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..activeLearner import ActiveLearner


class BoundaryPairSampler(ActiveLearner):
    def append(self, indices, x, distance_dict, Y_joined):
        if Y_joined[indices[x][0]] is Y_joined[indices[x][1]]:
            return False  # limit to distance between different classes
        if (indices[x][1], indices[x][0]) in distance_dict:
            return False  # avoid double point pairs
        return True

    def calculate_next_query_indices(self, X_train_unlabeled_cluster_indices):
        print("BoundaryPairSampler is broken. Please fix or move on.")
        exit(-2)
        Y_pred = self.clf_list[0].predict(self.X_train_unlabeled)

        Y_joined = np.append(Y_pred, self.Y_train_labeled0)
        X_joined = np.append(self.X_train_unlabeled, self.X_train_labeled0)
        index_joined = np.indices(X_joined)

        nbrs = NearestNeighbors(
            n_neighbors=2, algorithm="ball_tree", n_jobs=self.config.cores
        ).fit(X_joined)
        distances, indices = nbrs.kneighbors(X_joined)

        distance_dict = {}

        for x in range(len(distances)):
            if self.append(indices, x, distance_dict, Y_joined) is True:
                distance_dict[(indices[x][0], indices[x][1])] = distances[x][1]

        myset = []

        while len(myset) < self.nr_queries_per_iteration:
            if len(distance_dict.keys()) == 0:
                pprint(distances)
                pprint(indices)

            # somehow distance_dict is empty?!
            index_pair = min(distance_dict, key=distance_dict.get)

            if index_pair[0] < len(self.X_train_unlabeled):
                true_index = index_joined[index_pair[0]]
                myset.append(true_index)

            if index_pair[1] < len(self.X_train_unlabeled):
                true_index = index_joined[index_pair[1]]
                myset.append(true_index)

            # myset = np.unique(myset)
            del distance_dict[index_pair]

        return myset
