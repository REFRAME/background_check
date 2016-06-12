from __future__ import division
import numpy as np

from sklearn.svm import SVC
from scipy.special import expit
import copy
from scipy.stats import norm

from background_check import BackgroundCheck


class OcDecomposition(object):
    def __init__(self, base_estimator=BackgroundCheck(),
                 normalization=None):
        self._base_estimator = base_estimator
        self._estimators = []
        self._thresholds = []
        self._normalization = normalization
        self._priors = []
        self._means = []

    def fit(self, X, y, threshold_percentile=10, mus=None, ms=None):
        classes = np.unique(y)
        n_classes = np.alen(classes)
        class_count = np.bincount(y)
        self._priors = class_count / np.alen(y)
        for c_index in np.arange(n_classes):
            c = copy.deepcopy(self._base_estimator)
            c.fit(X[y == c_index])
            self._estimators.append(c)
        scores = self.score(X, mus=mus, ms=ms)
        self._thresholds = np.zeros(len(self._estimators))
        for c_index in np.arange(n_classes):
            u = np.unique(scores[:, c_index])
            self._thresholds[c_index] = np.percentile(u, threshold_percentile)
        # self._thresholds = np.percentile(scores, threshold_percentile, axis=0)
        # for i, t in enumerate(self._thresholds):
        #     if t == 0.0:
        #         s = scores[:, i]
        #         self._thresholds[i] = np.amin(s[s > 0])
        self._means = scores.mean(axis=0)

    def score(self, X, mus=None, ms=None):
        if type(self._base_estimator) is BackgroundCheck:
            return self.score_bc(X, mus=mus, ms=ms)
        elif self._normalization in ["O-norm", "T-norm"]:
            return self.score_dens(X) + 1e-8    # this value is added to avoid
                                                # having 0-valued thresholds,
                                                # which is a problem for o-norm

    def score_dens(self, X):
        n = np.alen(X)
        scores = np.zeros((n, len(self._estimators)))
        for i, estimator in enumerate(self._estimators):
            s = np.exp(estimator.score(X))
            scores[range(n), i] = s
        return scores

    def score_bc(self, X, mus=None, ms=None):
        n = np.alen(X)
        probas = np.zeros((n, len(self._estimators)))
        for i, estimator in enumerate(self._estimators):
            if mus is None:
                mu = None
            else:
                mu = mus[i]
            if ms is None:
                m = None
            else:
                m = ms[i]
            probas[range(n), i] = estimator.predict_proba(X, mu=mu, m=m)[:, 1]
        return probas

    def predict(self, X, mus=None, ms=None):
        scores = self.score(X, mus=mus, ms=ms)
        if type(self._base_estimator) is BackgroundCheck:
            return self.predict_bc(scores)
        elif self._normalization == "O-norm":
            return self.predict_o_norm(scores)
        elif self._normalization == "T-norm":
            return self.predict_t_norm(scores)

    def predict_o_norm(self, scores):
        reject = scores <= self._thresholds
        scores /= self._thresholds
        scores[reject] = -1
        max_scores = scores.max(axis=1)
        predictions = scores.argmax(axis=1)
        predictions[max_scores <= 1] = len(self._estimators)
        return predictions

    def predict_t_norm(self, scores):
        reject = scores <= self._thresholds
        scores -= self._thresholds
        means = self._means - self._thresholds
        scores = (scores / means) * self._priors
        scores[reject] = -np.inf
        max_scores = scores.max(axis=1)
        predictions = scores.argmax(axis=1)
        predictions[max_scores <= 0] = len(self._estimators)
        return predictions

    def predict_bc(self, scores):
        reject = scores <= self._thresholds
        total_reject = (np.sum(reject, axis=1) == len(self._estimators))
        scores[reject] = -1
        predictions = scores.argmax(axis=1)
        predictions[total_reject] = len(self._estimators)
        return predictions

    def accuracy(self, X, y, mus=None, ms=None):
        predictions = self.predict(X, mus=mus, ms=ms)
        return np.mean(predictions == y)

    @property
    def thresholds(self):
        return self._thresholds
