import random
from collections import defaultdict
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import abc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import (DBSCAN, OPTICS, AgglomerativeClustering, Birch,
                             KMeans)
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


class BaseClusterStrategy:
    def plot_dendrogram(self, **kwargs):
        self.cluster_model.fit(self.X_train_combined_pca)
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
        y_pred = self.cluster_model.fit_predict(self.X_train_combined_pca,
                                                self.Y_train_labeled)

        # plot the top three levels of the dendrogram
        plt.figure()
        dimension = 0
        plt.scatter(self.X_train_combined_pca[:, dimension],
                    self.X_train_combined_pca[:, dimension + 1],
                    c=y_pred)

        plt.show()

    def set_data(self, X_train_labeled, Y_train_labeled, X_train_unlabeled,
                 label_encoder):
        # first run pca to downsample data
        self.X_train_combined = np.concatenate(
            (X_train_labeled, X_train_unlabeled))
        self.X_train_unlabeled = X_train_unlabeled
        self.Y_train_labeled = self.Y_train_labeled

        n_clusters = len(label_encoder.classes_)

        #  n_clusters = 8
        self.pca = PCA(n_components=n_clusters)
        self.pca.fit(self.X_train_combined)
        self.X_train_combined_pca = self.pca.transform(self.X_train_combined)
        # then cluster it
        self.cluster_model = AgglomerativeClustering(
            n_clusters=int(self.X_train_combined.shape[1] / 10)
        )  #distance_threshold=0,                                                     n_clusters=None)
        #  self.plot_cluster()
        #  self.plot_dendrogram()

    @abc.abstractmethod
    def get_oracle_cluster(self):
        # return X_train_unlabeled
        pass

    @abc.abstractmethod
    def get_global_query_indice(self, cluster_query_indices):
        # return global_query_indices
        pass
