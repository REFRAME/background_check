from __future__ import division
import numpy as np

from sklearn.svm import SVC
from scipy.special import expit
import copy


class OvoClassifier(object):
    def __init__(self, base_classifier=SVC(kernel='linear')):
        self._base_classifier = base_classifier
        self._classifiers = []
        self._combinations = []
        self._n_classes = 0

    def fit(self, X, y):
        classes = np.unique(y)
        self._n_classes = np.alen(classes)
        self._combinations = np.array([[c1, c2] for i, c1 in enumerate(classes)
                                      for c2 in classes[i+1:]])
        for index, combination in enumerate(self._combinations):
            c = copy.deepcopy(self._base_classifier)
            indices = np.in1d(y, combination)
            y_train = y[indices]
            c.fit(X[indices], (y_train == combination[1]).astype(int))
            self._classifiers.append(c)

    def predict(self, X):
        n = np.alen(X)
        votes = np.zeros((n, self._n_classes))
        confidences = np.ones((n, self._n_classes))*2
        for index, combination in enumerate(self._combinations):
            predictions = combination[self._classifiers[index].predict(X)]
            votes[range(n), predictions] = 1
            value = expit(self._classifiers[index].decision_function(X))
            confidences[range(n), predictions] = np.minimum(confidences[
                                                                range(n),
                                                                predictions],
                                                            value)
        predictions = votes.argmax(axis=1)
        confidences = confidences[range(n), predictions]
        return predictions, confidences
