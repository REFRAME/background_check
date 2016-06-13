from __future__ import division
import numpy as np
import copy

from sklearn.svm import SVC
from sklearn.metrics import log_loss

from scipy.optimize import minimize

from ovo_classifier import OvoClassifier
from confident_classifier import ConfidentClassifier


class Ensemble(object):
    def __init__(self, base_classifier=OvoClassifier(), n_ensemble=1,
                 bootstrap_percent=0.75, lambd=0.0):
        self._base_classifier = base_classifier
        self._classifiers = []
        self._weights = []
        self._n_ensemble = n_ensemble
        self._percent = bootstrap_percent
        self._lambda = lambd

    def fit(self, X, y, xs=None, ys=None):
        init = xs is None
        if init:
            xs = []
            ys = []
        for c_index in np.arange(self._n_ensemble):
            if init:
                x_train, y_train = bootstrap(X, y, self._percent)
                xs.append(x_train)
                ys.append(y_train)
            else:
                x_train = xs[c_index]
                y_train = ys[c_index]
            c = copy.deepcopy(self._base_classifier)
            c.fit(x_train, y_train, np.alen(np.unique(y)))
            self._classifiers.append(c)
        self.prune_ensemble(X, y)
        return xs, ys

    def prune_ensemble_old(self, X, y):
        predictions, confidences = self.get_weights(X, y)
        sorted_indices = np.argsort(1.0/(self._weights + 1.0))
        n_classes = np.alen(np.unique(y))
        n = np.alen(X)
        accuracies = np.zeros(self._n_ensemble+1)
        for j in np.arange(self._n_ensemble+1):
            votes = np.zeros((n, n_classes))
            for c_index in np.arange(0, j):
                i = sorted_indices[c_index]
                pred = predictions[:, i]
                conf = confidences[:, i]
                votes[range(n), pred] += conf * self._weights[i]
            if not np.all(votes == 0.0):
                accuracies[j] = np.mean(votes.argmax(axis=1) == y)
        final_j = np.argmax(accuracies)
        self._classifiers = [self._classifiers[sorted_indices[i]] for i in
                           range(final_j)]
        self._weights = self._weights[sorted_indices[:final_j]]
        self._n_ensemble = final_j

    def prune_ensemble(self, X, y):
        predictions, confidences = self.get_weights(X, y)
        sorted_indices = np.argsort(1.0/(self._weights + 1.0))
        n_classes = np.alen(np.unique(y))
        n = np.alen(X)
        accuracies = np.zeros(self._n_ensemble)
        votes = np.zeros((n, n_classes))
        for c_index in np.arange(self._n_ensemble):
            i = sorted_indices[c_index]
            pred = predictions[:, i]
            conf = confidences[:, i]
            votes[range(n), pred] += conf * self._weights[i]
            # if not np.all(votes == 0.0):
            accuracies[c_index] = np.mean(votes.argmax(axis=1) == y)
        final_j = np.argmax(accuracies) + 1
        self._classifiers = [self._classifiers[sorted_indices[i]] for i in
                             range(final_j)]
        self._weights = self._weights[sorted_indices[:final_j]]
        self._n_ensemble = final_j

    def get_weights(self, X, y):
        if type(self._base_classifier) is not ConfidentClassifier\
           and self._base_classifier.classifier_type is not ConfidentClassifier:
            return self.get_weights_li(X, y)
        else:
            return self.get_weights_bc(X, y)

    def get_weights_li(self, X, y):
        n = np.alen(y)
        predictions = np.zeros((n, self._n_ensemble))
        confidences = np.zeros((n, self._n_ensemble))
        margins = np.zeros((n, self._n_ensemble))
        for c_index, c in enumerate(self._classifiers):
            res = c.predict(X)
            predictions[:, c_index] = res[0]
            marg = (res[0] == y).astype(float)
            marg[marg == 0] = -1
            margins[:, c_index] = marg
            confidences[:, c_index] = res[1]
        # conf_pred = predictions * confidences
        marg_pred = margins * confidences
        # f = lambda w: np.sum(np.power(1.0 - y*(w*conf_pred).sum(axis=1),
        #                               2.0)) + self._lambda*np.linalg.norm(w)
        f = lambda w: np.sum(np.power(1.0 - (w*marg_pred).sum(axis=1),
                                      2.0)) + self._lambda*np.linalg.norm(w)
        w0 = np.ones(self._n_ensemble)/self._n_ensemble
        cons = ({'type': 'eq', 'fun': lambda w:  1.0 - np.sum(w)})

        bounds = [(0, 1) for c_index in range(self._n_ensemble)]
        res = minimize(f, w0, bounds=bounds, constraints=cons)
        self._weights = res.x
        return predictions.astype(int), confidences

    def get_weights_bc(self, X, y):
        n = np.alen(y)
        predictions = np.zeros((n, self._n_ensemble))
        confidences = np.zeros((n, self._n_ensemble))
        checks = np.zeros(self._n_ensemble)
        for c_index, c in enumerate(self._classifiers):
            res = get_predictions(c, X)
            predictions[:, c_index] = res[0]
            confidences[:, c_index] = res[1]
            checks[c_index] = np.mean(res[2])
        self._weights = checks / np.sum(checks)
        return predictions.astype(int), confidences

    def predict(self, X):
        n = np.alen(X)
        for c_index, c in enumerate(self._classifiers):
            res = get_predictions(c, X)
            pred = res[0]
            conf = res[1]
            if c_index == 0:
                votes = np.zeros((n, c.n_classes))
            votes[range(n), pred] += conf * self._weights[c_index]
        return votes.argmax(axis=1)

    def predict_proba(self, X):
        n = np.alen(X)
        for c_index, c in enumerate(self._classifiers):
            res = get_predictions(c, X)
            pred = res[0]
            conf = res[1]
            if c_index == 0:
                votes = np.zeros((n, c.n_classes))
            votes[range(n), pred] += conf * self._weights[c_index]
        proba = votes/votes.sum(axis=1).reshape(-1,1)
        proba[np.isnan(proba)] = 1/proba.shape[1]
        return proba

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def log_loss(self, X, y):
        y_pred = self.predict_proba(X)
        return log_loss(y, y_pred, eps=1e-3)


def get_predictions(c, X):
    if type(c) is OvoClassifier:
        return c.predict(X)
    elif type(c) is ConfidentClassifier:
        n = np.alen(X)
        probas = c.predict_proba(X)
        predictions = np.argmax(probas[:, :-1], axis=1)
        confidences = probas[range(n), predictions]
        checks = 1.0 - probas[:, -1]
        return [predictions, confidences, checks]


def bootstrap(x, y, percent):
    n = np.alen(x)
    indices = np.random.choice(n, int(np.around(n * percent)))
    return x[indices], y[indices]

