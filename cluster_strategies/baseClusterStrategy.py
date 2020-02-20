import logging
import abc
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import (DBSCAN, OPTICS, AgglomerativeClustering, Birch,
                             KMeans)
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from sklearn import metrics
from math import log, e


class BaseClusterStrategy:
    def _entropy(self, labels):
        n_labels = len(labels)
        if n_labels <= 1:
            return 0
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / n_labels
        n_classes = np.count_nonzero(probs)

        if n_classes <= 1:
            return 0

        ent = 0

        # compute entropy
        base = e
        for i in probs:
            ent -= i * log(i, base)
        return ent

    def set_data_storage(self, data_storage):
        self.data_storage = data_storage
        # first run pca to downsample data

        X_train_combined = pd.concat([
            self.data_storage.X_train_labeled,
            self.data_storage.X_train_unlabeled
        ])
        self.X_train_combined = X_train_combined

        # then cluster it
        self.cluster_model = AgglomerativeClustering(
            n_clusters=int(X_train_combined.shape[1] / 5)
        )  #distance_threshold=0,                                                     n_clusters=None)
        #  self.plot_cluster()
        #  self.plot_dendrogram()

        self.data_storage = data_storage

        # fit cluster
        self.Y_train_unlabeled_cluster = self.cluster_model.fit_predict(
            self.data_storage.X_train_unlabeled)

        logging.info("Clustering into " + str(self.cluster_model.n_clusters) +
                     " cluster")

        self.data_storage.X_train_unlabeled_cluster_indices = defaultdict(
            lambda: list())
        self.data_storage.X_train_labeled_cluster_indices = defaultdict(
            lambda: list())

        for cluster_index, X_train_index in zip(
                self.Y_train_unlabeled_cluster,
                self.data_storage.X_train_unlabeled.index):
            self.data_storage.X_train_unlabeled_cluster_indices[
                cluster_index].append(X_train_index)

        #  for cluster_index, X_train_indices in self.data_storage.X_train_unlabeled_cluster_indices.items(
        #  ):
        #  cluster_labels = self.data_storage.Y_train_unlabeled.loc[
        #  X_train_indices][0].to_numpy()
        #  logging.info(self._entropy(cluster_labels), '\t', cluster_labels)

    def plot_dendrogram(self, **kwargs):
        self.cluster_model.fit(self.X_train_combined)
        model = self.cluster_model
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)
        plt.show()

    def plot_cluster(self):
        y_pred = self.cluster_model.fit_predict(
            self.X_train_combined, self.data_storage.Y_train_labeled)

        # plot the top three levels of the dendrogram
        plt.figure()
        dimension = 0
        plt.scatter(self.X_train_combined[:, dimension],
                    self.X_train_combined[:, dimension + 1],
                    c=y_pred)

        plt.show()

    @abc.abstractmethod
    def get_cluster_indices(self, **kwargs):
        # return X_train_unlabeled
        pass
