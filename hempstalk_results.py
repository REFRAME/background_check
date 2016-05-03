from __future__ import division
import numpy as np
from sklearn.mixture import GMM
from sklearn.metrics import roc_auc_score
np.random.seed(42)
import matplotlib.pyplot as plt
plt.ion()
plt.rcParams['figure.figsize'] = (7,4)
plt.rcParams['figure.autolayout'] = True

from sklearn.cross_validation import StratifiedKFold

from cwc.synthetic_data.datasets import MLData


mldata = MLData()


def separate_sets(x, y, test_fold, test_folds):
    test_data = x[test_folds == test_fold, :]
    test_labels = y[test_folds == test_fold]

    training_data = x[test_folds != test_fold, :]
    training_labels = y[test_folds != test_fold]
    return [training_data, training_labels, test_data, test_labels]


for name, dataset in mldata.datasets.iteritems():
    print name
    print dataset.classes
    weighted_auc = 0.0
    mc_iterations = 10.0
    n_folds = 10.0
    counts = dataset.counts
    for mc in np.arange(mc_iterations):
        skf = StratifiedKFold(dataset.target, n_folds=n_folds, shuffle=True)
        test_folds = skf.test_folds
        for test_fold in np.arange(n_folds):
            training_data, training_labels, test_data, test_labels = \
             separate_sets(dataset.data, dataset.target, test_fold, test_folds)
            n_training = np.alen(training_labels)
            for actual_class in dataset.classes:
                if counts[actual_class] >= 2*n_folds:
                    tr_class = training_data[training_labels == actual_class, :]
                    t_labels = (test_labels == actual_class).astype(int)
                    g = GMM()
                    g.fit(tr_class)
                    scores = g.score(test_data)
                    auc = roc_auc_score(t_labels, scores)
                    prior = np.alen(tr_class) / n_training
                    weighted_auc += auc*prior
    weighted_auc /= (n_folds * mc_iterations)
    print ("Weighted AUC: {}".format(weighted_auc))
