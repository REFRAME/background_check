from __future__ import division
import numpy as np

from sklearn.svm import SVC
from scipy.special import expit
import copy

from confident_classifier import ConfidentClassifier


class OvoClassifier(object):
    def __init__(self, base_classifier=SVC(kernel='linear')):
        self._base_classifier = base_classifier
        self._classifiers = []
        self._combinations = []
        self._n_classes = 0

    def fit(self, X, y, total_classes=0):
        classes = np.unique(y)
        n_classes = np.alen(classes)
        if total_classes > 0:
            self._n_classes = total_classes
        else:
            self._n_classes = n_classes
        self._combinations = np.array([[c1, c2] for i, c1 in enumerate(classes)
                                      for c2 in classes[i+1:]])
        for index, combination in enumerate(self._combinations):
            c = copy.deepcopy(self._base_classifier)
            indices = np.in1d(y, combination)
            y_train = y[indices]
            c.fit(X[indices], (y_train == combination[1]).astype(int))
            self._classifiers.append(c)

    def predict_proba(self, X):
        n = np.alen(X)
        confidences = np.ones((n, self._n_classes))
        for index, combination in enumerate(self._combinations):
            probas = self._classifiers[index].predict_proba(X)
            old_conf = confidences[:, combination]
            confidences[:, combination] = old_conf * probas
        return confidences / (confidences.sum(axis=1).reshape(-1, 1))

    def predict(self, X, mu=None, m=None):
        if type(self._base_classifier) is ConfidentClassifier:
            return self.predict_bc(X, mu=mu, m=m)
        else:
            return self.predict_non_bc(X)

    def predict_non_bc(self, X):
        n = np.alen(X)
        votes = np.zeros((n, self._n_classes))
        confidences = np.ones((n, self._n_classes))*2
        for index, combination in enumerate(self._combinations):
            predictions = combination[self._classifiers[index].predict(X)]
            votes[range(n), predictions] += 1
            value = expit(self._classifiers[index].decision_function(X))
            confidences[range(n), predictions] = np.minimum(confidences[
                                                                range(n),
                                                                predictions],
                                                            value)
        predictions = votes.argmax(axis=1)
        confidences = confidences[range(n), predictions]
        return [predictions, confidences]

    def predict_bc(self, X, mu=None, m=None):
        n = np.alen(X)
        votes = np.zeros((n, self._n_classes))
        confidences = np.ones((n, self._n_classes))*2
        check_probs = np.ones((n, self._n_classes))*2
        for index, combination in enumerate(self._combinations):
            probas = self._classifiers[index].predict_proba(X, mu=mu, m=m)
            winners = np.argmax(probas[:, :-1], axis=1)
            predictions = combination[winners]
            votes[range(n), predictions] += 1
            old_conf = confidences[range(n), predictions]
            new_conf = probas[range(n), winners]
            confidences[range(n), predictions] = np.minimum(old_conf, new_conf)

            old_check = confidences[range(n), predictions]
            new_check = 1.0 - probas[:, -1]
            check_probs[range(n), predictions] = np.minimum(old_check,
                                                            new_check)

        predictions = votes.argmax(axis=1)
        confidences = confidences[range(n), predictions]
        check_probs = check_probs[range(n), predictions]
        return [predictions, confidences, check_probs]

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def classifier_type(self):
        return type(self._base_classifier)
