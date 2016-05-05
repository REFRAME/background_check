from __future__ import division
import numpy as np
from sklearn.mixture import GMM
from sklearn.metrics import roc_auc_score
from scipy.stats import multivariate_normal
np.random.seed(42)
import matplotlib.pyplot as plt
plt.ion()
plt.rcParams['figure.figsize'] = (7,4)
plt.rcParams['figure.autolayout'] = True

from sklearn.cross_validation import StratifiedKFold

from cwc.synthetic_data.datasets import MLData


mldata = MLData()

class MyMultivariateNormal(object):
    def __init__(self, min_covar=0.001, covariance_type='diag'):
        self.min_covar = min_covar
        self.covariance_type = covariance_type
        self.alpha = 0.00000001


        if covariance_type not in ['full', 'diag',]:
            raise ValueError('Invalid value for covariance_type: %s' %
                             covariance_type)

    def pseudo_determinant(self, A, alpha):
        n = len(A)
        return np.linalg.det(A + np.eye(n)*alpha)/ np.power(alpha, n-np.rank(A))

    def fit(self, x):
        self.mu = x.mean(axis=0)
        self.sigma = np.cov(x.T)
        self.sigma[self.sigma==0] = self.min_covar
        if(self.covariance_type == 'diag'):
            self.sigma = np.eye(np.alen(self.sigma))*self.sigma

        self.size = x.shape[1]

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


def separate_sets(x, y, test_fold_id, test_folds):
    x_test = x[test_folds == test_fold_id, :]
    y_test = y[test_folds == test_fold_id]

    x_train = x[test_folds != test_fold_id, :]
    y_train = y[test_folds != test_fold_id]
    return [x_train, y_train, x_test, y_test]


class MultivariateNormal(object):
    def __init__(self, allow_singular=True):
        self.allow_singular = allow_singular

    def fit(self, x):
        self.mu = x.mean(axis=0)
        self.sigma = np.cov(x.T)
        self.model = multivariate_normal(mean=self.mu, cov=self.sigma,
                allow_singular=self.allow_singular)

    def score(self,x):
        return self.model.pdf(x)

mc_iterations = 10.0
n_folds = 10.0
weighted_auc = np.full((len(mldata.datasets), mc_iterations, n_folds), -1)

for i, (name, dataset) in enumerate(mldata.datasets.iteritems()):
    print i
    mldata.sumarize_datasets(name)
    for mc in np.arange(mc_iterations):
        skf = StratifiedKFold(dataset.target, n_folds=n_folds, shuffle=True)
        test_folds = skf.test_folds
        for test_fold in np.arange(n_folds):
            x_train, y_train, x_test, y_test = separate_sets(
                    dataset.data, dataset.target, test_fold, test_folds)
            n_training = np.alen(y_train)
            w_auc_fold = 0
            prior_sum = 0
            for actual_class in dataset.classes:
                tr_class = x_train[y_train == actual_class, :]
                t_labels = (y_test == actual_class).astype(int)
                prior = np.alen(tr_class) / n_training
                prior_sum += prior
                if np.alen(tr_class) > 1 and not all(t_labels == 0):
                    n_c = tr_class.shape[1]
                    if n_c > np.alen(tr_class):
                        n_c = np.alen(tr_class)
                    #model = GMM(n_components=n_c,
                    #            covariance_type='diag')
                    #model = GMM(n_components=1, covariance_type='full',
                    #        verbose=2)
                    model = MyMultivariateNormal(covariance_type='diag')
                    #model = MultivariateNormal()

                    model.fit(tr_class)
                    scores = model.score(x_test)

                    auc = roc_auc_score(t_labels, scores)
                    w_auc_fold += auc*prior
            weighted_auc[i, mc, test_fold] = w_auc_fold / prior_sum
    w_auc_mean = weighted_auc[i].mean()
    w_auc_std = weighted_auc[i].std()
    print ("Weighted AUC: {} +- {}".format(w_auc_mean, w_auc_std))



print("dataset,size,features,classes,wauc mean, wauc std,min,max")
for i, (name, dataset) in enumerate(mldata.datasets.iteritems()):
    w_auc_mean = weighted_auc[i].mean()
    w_auc_std = weighted_auc[i].std()
    size = dataset.data.shape[0]
    n_features = dataset.data.shape[1]
    classes = len(dataset.classes)
    minimum = dataset.data.min()
    maximum = dataset.data.max()
    if w_auc_mean != -1:
        print ("{},{},{},{},{},{},{},{}".format(name, size, n_features,
            classes, w_auc_mean, w_auc_std, minimum, maximum))
