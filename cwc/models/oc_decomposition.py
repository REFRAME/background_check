from __future__ import division
import numpy as np

from sklearn.svm import SVC
from scipy.special import expit
import copy

from background_check import BackgroundCheck


class OcDecomposition(object):
    def __init__(self, base_estimator=BackgroundCheck(),
                 normalization=None):
        self._base_estimator = base_estimator
        self._estimators = []
        self._thresholds = []
        self._normalization = normalization

    def fit(self, X, y, threshold_percentile=10):
        classes = np.unique(y)
        n_classes = np.alen(classes)
        for c_index in np.arange(n_classes):
            c = copy.deepcopy(self._base_estimator)
            c.fit(X[y == c_index])
            self._estimators.append(c)
        scores = self.score(X)
        self._thresholds = np.percentile(scores, threshold_percentile, axis=0)

    def score(self, X):
        if type(self._base_estimator) is BackgroundCheck:
            return self.score_bc(X)
        elif self._normalization in ["O-norm", "T-norm"]:
            return self.score_dens(X) + 1e-8

    def score_dens(self, X):
        n = np.alen(X)
        scores = np.zeros((n, len(self._estimators)))
        for i, estimator in enumerate(self._estimators):
            s = np.exp(estimator.score(X))
            if np.alen(s) == 1:
                s = np.exp(estimator.score_samples(X))
            scores[range(n), i] = s
        return scores

    def score_bc(self, X):
        n = np.alen(X)
        probas = np.zeros((n, len(self._estimators)))
        for i, estimator in enumerate(self._estimators):
            probas[range(n), i] = estimator.predict_proba(X)[:, 1]
        return probas

    def predict(self, X):
        scores = self.score(X)
        if type(self._base_estimator) is BackgroundCheck:
            return self.predict_bc(scores)
        elif self._normalization == "O-norm":
            return self.predict_o_norm(scores)

    def predict_o_norm(self, scores):
        scores /= self._thresholds
        max_scores = scores.max(axis=1)
        predictions = scores.argmax(axis=1)
        predictions[max_scores <= 1] = len(self._estimators)
        return predictions

    def predict_t_norm(self, scores):
        scores /= self._thresholds
        max_scores = scores.max(axis=1)
        predictions = scores.argmax(axis=1)
        predictions[max_scores <= 1] = len(self._estimators)
        return predictions

    def predict_bc(self, scores):
        reject = np.sum(scores <= self._thresholds, axis=1)
        reject = (reject == len(self._estimators))
        predictions = scores.argmax(axis=1)
        predictions[reject] = len(self._estimators)
        return predictions

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    @property
    def thresholds(self):
        return self._thresholds
