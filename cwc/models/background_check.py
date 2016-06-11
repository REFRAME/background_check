from __future__ import division
import numpy as np
from sklearn.mixture import GMM
from sklearn import datasets
from scipy.special import expit
from sklearn.svm import OneClassSVM


class BackgroundCheck(object):
    """This class is responsible for performing background checks.

    Args:
        estimator (object): An object of a class that performs density
            estimation. Any density estimator can be used, as long as it has
            the following methods: fit(X) and score(X) or decision_function(X),
            where X is an array with shape = [n_samples, n_features].
        mu (float): Parameter that sets the deepness of the background
            valleys, compared to foreground peaks.
        m (float): Parameter for how high is the uniform background, compared
            to foreground peaks.

    Attributes:
        _max_dens (float): Maximum density value estimated from training data.

    """
    def __init__(self, estimator=GMM(n_components=1, covariance_type='diag'),
                 mu=0.0, m=1.0):
        self._estimator = estimator
        self._mu = mu
        self._m = m
        self._max_dens = 0.0
        self._delta = 0.0

    def fit(self, X):
        """Fits the density estimator to the data in X.

        Args:
            X (array-like, shape = [n_samples, n_features]): training data.

        Returns:
            Nothing.

        """
        self._estimator.fit(X)
        dens = self.score(X)
        self._delta = 0.0 - dens.min()
        dens = expit(dens + self._delta)
        if hasattr(self._estimator, 'maximum'):
            self._max_dens = expit(self._estimator.maximum[0] + self._delta)
        else:
            self._max_dens = dens.max()

    def predict_proba(self, X, mu=None, m=None):
        """Performs background check on the data in X.

        Args:
            X (array-like, shape = [n_samples, n_features]): training data.

        Returns:
            posteriors (array-like, shape = [n_samples, 2]): posterior
            probabilities for background (column 0) and foreground (column 1).

        """
        if mu is None:
            mu = self._mu
        if m is None:
            m = self._m
        q, p_x_and_b = self.compute_q_p_x_and_b(X, mu=mu, m=m)
        posteriors = np.zeros((np.alen(X), 2))
        posteriors[:, 0] = p_x_and_b / (p_x_and_b + q)
        posteriors[:, 1] = 1.0 - posteriors[:, 0]
        return posteriors

    def compute_q_p_x_and_b(self, X, mu=None, m=None):
        """

        Returns:
            q and not q
        """
        if mu is None:
            mu = self._mu
        if m is None:
            m = self._m
        # TODO maybe apply expit only on the necessary cases
        p_x_and_f = expit(self.score(X) + self._delta)
        # TODO look for other methods to clip the probabilities?
        q = np.clip(p_x_and_f / self._max_dens, 0.0, 1.0)
        p_x_and_b = q * mu + (1.0 - q) * m
        return q, p_x_and_b

    def score(self, X):
        """Gets scores for the objects of X using different functions that
        depend on the estimator.

        Args:
            X (array-like, shape = [n_samples, n_features]): training data.

        Returns:
            (array-like, shape = [n_samples]): scores given by the estimator.

        """
        if 'score' in dir(self._estimator):
            s = self._estimator.score(X)
            if np.alen(s) == 1:
                s = self._estimator.score_samples(X)
            return s
        elif 'decision_function' in dir(self._estimator):
            return self._estimator.decision_function(X).reshape(-1)


def test():
    np.random.seed(42)
    dataset = datasets.load_iris()

    test_indices = np.random.choice(150, 10)
    test_set = dataset.data[test_indices, :]
    test_set = np.vstack((test_set, np.array([[0.0, 0.0, 0.0, 0.0]])))
    test_set = np.vstack((test_set, np.array([[10.0, 10.0, 10.0, 10.0]])))

    print "GMM"
    gmm = GMM(n_components=3, covariance_type='diag')
    bc_gmm = BackgroundCheck(estimator=gmm, mu=0.0, m=1.0)
    bc_gmm.fit(dataset.data)
    print bc_gmm.predict_proba(test_set)

    print "OneClassSVM"
    sv = OneClassSVM()
    bc_sv = BackgroundCheck(estimator=sv, mu=0.0, m=1.0)
    bc_sv.fit(dataset.data)
    print bc_sv.predict_proba(test_set)


if __name__ == "__main__":
    pass
