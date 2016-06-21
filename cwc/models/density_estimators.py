from sklearn import svm
from ..data_wrappers import reject
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GMM
from sklearn.neighbors import KernelDensity


class DensityEstimators(object):
    def __init__(self):
        self.models = {}
        self.unknown = {}
        self.known = {}

    def train_confidence_model(self, X_kno, X_unk):
        """Train a classifier of training points

        Returns a classifier that predicts high probability values for training
        points and low probability values for reject points.
        """
        model = svm.SVC(probability=True)
        #model = tree.DecisionTreeClassifier(max_depth=5)

        X_kno_unk = np.vstack((X_kno,X_unk))
        y = np.hstack((np.ones(np.alen(X_kno)), np.zeros(np.alen(X_unk)))).T
        model.fit(X_kno_unk, y)

        return model

    def _train_aggregation_model(self, X):
        scores_kno = self.predict_proba(X)
        self.scores_agg_unk = reject.create_reject_data(scores_kno,
                proportion=1, method='uniform_hsphere', pca=True,
                pca_variance=0.99, pca_components=0, hshape_cov=0,
                hshape_prop_in=0.99, hshape_multiplier=1.5)
        model_agg = self.train_confidence_model(scores_kno,self.scores_agg_unk)

        return model_agg

    def train(self,X,Y):
        """
            TODO for PCA to work we need more instances than features, if we
            reduce the problem to M binary subproblems the number of instances
            is reduced by M while the number of features remains constant.
            This can be a problem for MNIST, CIFAR and ImageNet.

        """
        self.classes = np.unique(Y)
        self.accuracies = {}
        for y in self.classes:
            x = X[Y==y]
            self.unknown[y] = reject.create_reject_data(x, proportion=1,
                        method='uniform_hsphere', pca=True, pca_variance=0.99,
                        pca_components=0, hshape_cov=0, hshape_prop_in=0.99,
                        hshape_multiplier=1.5)
            self.models[y] = self.train_confidence_model(x,self.unknown[y])

        self.model_agg = self._train_aggregation_model(X)

    def predict_proba(self,X):
        scores = np.zeros((np.alen(X), len(self.classes)))
        for index, y in enumerate(self.classes):
            scores[:,index] = self.models[y].predict_proba(X)[:,1]

        return scores

    def predict_confidence(self,X):
        scores = self.predict_proba(X)
        return self.model_agg.predict_proba(scores)[:,1]

class MyGMM(GMM):
    def score(self, X):
        return np.exp(super(MyGMM, self).score(X))

class MyMultivariateNormal(object):
    def __init__(self, mean=None, cov=None, min_covar=1e-10,
                 covariance_type='diag'):
        if mean is not None:
            self.mu = mean
            self.size = len(self.mu)
            # TODO assess that the parameters mean and cov are correct
            if cov is not None:
                # TODO create a function that computes deg, norm_const and inv
                self.sigma = cov
                self.det = np.linalg.det(self.sigma)
                self.norm_const = 1.0/ ( np.power((2*np.pi),float(self.size)/2) *
                        np.sqrt(self.det) )
                self.inv = np.linalg.inv(self.sigma)
        self.min_covar = min_covar
        self.covariance_type = covariance_type
        self.alpha = np.float32(1e-32)


        if covariance_type not in ['full', 'diag',]:
            raise ValueError('Invalid value for covariance_type: %s' %
                             covariance_type)

    def pseudo_determinant(self, A, alpha):
        n = len(A)
        return np.linalg.det(A + np.eye(n)*alpha)/ np.power(alpha, n-np.rank(A))

    def fit(self, x):
        self.mu = x.mean(axis=0)
        self.sigma = np.cov(x.T, bias=1) # bias=0 (N-1), bias=1 (N)
        self.sigma[self.sigma==0] = self.min_covar
        if(self.covariance_type == 'diag'):
            self.sigma = np.eye(np.alen(self.sigma))*self.sigma

        if len(self.mu.shape) == 0:
            self.size = 1
        else:
            self.size = self.mu.shape[0]

        self.det = np.linalg.det(self.sigma)
        # If sigma is singular
        if self.det == 0:
            self.pseudo_det = self.pseudo_determinant(self.sigma*2*np.pi, self.alpha)
            self.norm_const = 1.0/ np.sqrt(self.pseudo_det)
            self.inv = np.linalg.pinv(self.sigma)
        else:
            self.norm_const = 1.0/ ( np.power((2*np.pi),float(self.size)/2) *
                    np.sqrt(self.det) )
            self.inv = np.linalg.inv(self.sigma)

    def score(self,x):
        x_mu = np.subtract(x,self.mu)
        result = np.exp(-0.5 * np.diag(np.dot(x_mu,np.dot(self.inv,x_mu.T))))

        return self.norm_const * result

    # FIXME: look for an appropriate name
    def log_likelihood(self,x):
        x_mu = np.subtract(x,self.mu)
        result = -0.5 * np.diag(np.dot(x_mu,np.dot(self.inv,x_mu.T)))

        return self.norm_const * result

    @property
    def means_(self):
        return self.mu

    @property
    def covars_(self):
        if self.covariance_type == 'diag':
            return np.diag(self.sigma)
        return self.sigma

    def sample(self, n):
        return np.random.multivariate_normal(self.mu, self.sigma, n)

    @property
    def maximum(self):
        return self.score(np.array(self.mu).reshape(-1,1))


class MultivariateNormal(object):
    def __init__(self, mean=None, cov=None, allow_singular=True,
                 covariance_type='diag'):
        if mean is not None:
            self.mu = mean
        if cov is not None:
            self.sigma = cov
        self.allow_singular = allow_singular
        self.covariance_type = covariance_type

    def fit(self, x):
        self.mu = x.mean(axis=0)
        self.sigma = np.cov(x.T, bias=1) # bias=0 (N-1), bias=1 (N)
        if self.covariance_type == 'diag':
            self.sigma = np.eye(np.alen(self.sigma))*self.sigma

        self.model = multivariate_normal(mean=self.mu, cov=self.sigma,
                allow_singular=self.allow_singular)

    def score(self,x):
        return self.model.pdf(x)

    @property
    def means_(self):
        return self.mu

    @property
    def covars_(self):
        if self.covariance_type == 'diag':
            return np.diag(self.sigma)
        return self.sigma

    def sample(self, n):
        return np.random.multivariate_normal(self.mu, self.sigma, n)


class MyMultivariateKernelDensity(object):
    def __init__(self, kernel='gaussian', bandwidth=1.0):
        self._kernel = kernel
        self._bandwidth = bandwidth
        self._estimators = []

    def fit(self, X):
        p = X.shape[1]
        for feature in np.arange(p):
            kd = KernelDensity(kernel=self._kernel, bandwidth=self._bandwidth)
            kd.fit(X[:, feature].reshape(-1, 1))
            self._estimators.append(kd)

    def score(self, X):
        p = len(self._estimators)
        scores = np.zeros((np.alen(X), p))
        for feature in np.arange(p):
            s = self._estimators[feature].score_samples(
                X[:, feature].reshape(-1, 1))
            scores[:, feature] = s
        return scores.sum(axis=1)


