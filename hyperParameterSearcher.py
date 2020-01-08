import random
from abc import abstractmethod
from pprint import pprint
from shutil import rmtree
from tempfile import mkdtemp
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Memory
from scipy.stats import randint as sp_randint
from sklearn import svm
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (SelectFromModel, SelectKBest,
                                       VarianceThreshold, chi2)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_val_predict, train_test_split)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import (compute_class_weight,
                                        compute_sample_weight)

from experiment_setup_lib import load_and_prepare_X_and_Y, standard_config, evaluate

config = standard_config([(['--cache_size'], {
    'type': int,
    'help': "The amount of RAM to be used by SVM"
}), (['--nr_iteration'], {
    'type': int,
    'default': 5
})])


class HyperParameterSearcher:
    def __init__(self, config):
        self.config = config

    def loadDataset(self):
        X, Y, label_encoder = load_and_prepare_X_and_Y(config)

        # train/test split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X,
            Y,
            test_size=self.config.test_fraction,
            random_state=self.config.random_seed)

        self.X = X_train
        self.Y = Y_train
        self.label_encoder = label_encoder

    def testNaiveBayes(self):
        return {
            'alpha': [np.random.uniform(0.0, 1.0) for _ in range(100000)]
        }, MultinomialNB()

    def testSVMRbf(self):
        return {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf'],
            'cache_size': [self.config.cache_size],
        }, svm.SVC()

    def testSVMPoly(self):
        return {
            #  'degree': sp_randint(1, 10),
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['poly'],
            'cache_size': [self.config.cache_size],
        }, svm.SVC()

    def testDT(self):
        return {
            "criterion": ["gini", "entropy"],
            "max_depth": sp_randint(1, 50),
            "max_leaf_nodes": sp_randint(2, 50),
            "min_samples_leaf": sp_randint(1, 50),
            "min_samples_split": sp_randint(2, 50),
        }, DecisionTreeClassifier()

    def testRF(self):
        return {
            'criterion': ["gini", "entropy"],
            'n_estimators': sp_randint(5, 120),
            'max_features': [None, 'sqrt', 'log2'],
            'max_depth': sp_randint(1, 50),
            'min_samples_split': sp_randint(2, 50),
            'min_samples_leaf': sp_randint(1, 50),
            'max_leaf_nodes': sp_randint(2, 50),
        }, RandomForestClassifier()

    def find(self):
        if self.config.classifier == 'RF':
            param_distribution, clf = self.testRF()
        elif self.config.classifier == 'DTree':
            param_distribution, clf = self.testDT()
        elif self.config.classifier == 'SVMPoly':
            param_distribution, clf = self.testSVMPoly()
        elif self.config.classifier == 'SVMRbf':
            param_distribution, clf = self.testSVMRbf()
        elif self.config.classifier == 'NaiveBayes':
            param_distribution, clf = self.testNaiveBayes()

        grid = RandomizedSearchCV(
            clf,
            cv=5,
            n_jobs=self.config.cores,
            n_iter=self.config.nr_iteration,
            verbose=100,
            iid=False,
            param_distributions=param_distribution,
        )

        grid.fit(self.X,
                 self.Y,
                 sample_weight=compute_sample_weight('balanced', self.Y))

        clf = grid.best_estimator_

        evaluate(clf,
                 self.X,
                 self.Y,
                 self.config,
                 self.label_encoder,
                 store=True,
                 training_times=str(grid.best_params_) + "\n" +
                 str(grid.cv_results_) + "\n" + str(grid.best_score_))

    def classify(self):
        clf_params = functions.get_best_hyper_params('RF')

        clf = RandomForestClassifier(**clf_params)

        clf.fit(self.X,
                self.Y,
                sample_weight=compute_sample_weight('balanced', self.Y))

        yPredicted = cross_val_predict(clf, self.X, self.Y, cv=5)

        pprint(self.labelBinarizer.classes_)
        #
        #  get labels from labelbinarizer!!
        confusionMatrix = confusion_matrix(self.Y, yPredicted)

        classificationReport = classification_report(
            self.Y, yPredicted, target_names=self.labelBinarizer.classes_)

        print(classificationReport)
        print(confusionMatrix)


hyperParameterSearcher = HyperParameterSearcher(config)
hyperParameterSearcher.loadDataset()
hyperParameterSearcher.find()
