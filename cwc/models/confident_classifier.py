from __future__ import division
import numpy as np

from background_check import BackgroundCheck
from discriminative_models import MyDecisionTreeClassifier

from sklearn.mixture import GMM


class ConfidentClassifier(object):
    def __init__(self, classifier=MyDecisionTreeClassifier(), estimator=GMM(),
                 mu=0.0, m=1.0):
        self._classifier = classifier
        self._bc = BackgroundCheck(estimator=estimator, mu=mu, m=m)

    def fit(self, X, y):
        self._classifier.fit(X, y)
        self._bc.fit(X)

    def predict_proba(self, X, mu=None, m=None):
        class_posteriors = self._classifier.predict_proba(X)
        bc_posteriors = self._bc.predict_proba(X, mu=mu, m=m)
        class_posteriors = class_posteriors * bc_posteriors[:, 1].reshape(-1, 1)
        return np.hstack((class_posteriors, bc_posteriors[:, 0].reshape(-1, 1)))

    def predict_class_proba(self, X):
        return self._classifier.predict_proba(X)

    def predict_bc_proba(self, X, mu=None, m=None):
        return self._bc.predict_proba(X, mu=mu, m=m)

    def predict(self, X, mu=None, m=None):
        posteriors = self.predict_proba(X, mu, m)
        return np.argmax(posteriors, axis=1)
