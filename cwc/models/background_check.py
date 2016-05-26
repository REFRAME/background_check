from __future__ import division
import numpy as np
from sklearn.mixture import GMM
from sklearn import datasets


class BackgroundCheck(object):
    """This class is responsible for performing background checks.

    Args:
        estimator (object): An object of a class that performs density
            estimation. Any density estimator can be used, as long as it has
            the following methods: fit(X) and score(X), where X is an array
            with shape = [n_samples, n_features].
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

    def fit(self, X):
        """Fits the density estimator to the data in X.

        Args:
            X (array-like, shape = [n_samples, n_features]): training data.

        Returns:
            Nothing.

        """
        self._estimator.fit(X)
        self._max_dens = np.exp(self._estimator.score(X).max())

    def predict_proba(self, X):
        """Performs background check on the data in X.

        Args:
            X (array-like, shape = [n_samples, n_features]): training data.

        Returns:
            posteriors (array-like, shape = [n_samples, 2]): posterior
            probabilities for background (column 0) and foreground (column 1).

        """
        p_x_f = np.exp(self._estimator.score(X))
        posteriors = np.zeros((np.alen(X), 2))
        q = np.clip(p_x_f / self._max_dens, 0.0, 1.0)
        p_x_and_b = q * self._mu + (1.0 - q) * self._m
        posteriors[:, 0] = p_x_and_b / (p_x_and_b + q)
        posteriors[:, 1] = 1.0 - posteriors[:, 0]
        return posteriors


if __name__ == "__main__":
    np.random.seed(42)
    dataset = datasets.load_iris()
    print dataset.data.mean(axis=0)
    gmm = GMM(n_components=3, covariance_type='diag')
    bc = BackgroundCheck(estimator=gmm, mu=0.0, m=1.0)
    bc.fit(dataset.data)

    test_indices = np.random.choice(150, 10)
    test_set = dataset.data#[test_indices, :]
    test_set = np.vstack((test_set, np.array([[0.0, 0.0, 0.0, 0.0]])))
    test_set = np.vstack((test_set, np.array([[10.0, 10.0, 10.0, 10.0]])))
    print bc.predict_proba(test_set)
