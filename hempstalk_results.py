from __future__ import division
import numpy as np
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
    n_folds = 10
    skf = StratifiedKFold(dataset.target, n_folds=n_folds)
    for test_fold in np.arange(n_folds):
        training_data, training_labels, test_data, test_labels =
            separate_sets(dataset.data, dataset.target, test_fold, test_folds)
        for actual_class in dataset.classes:
            
