from __future__ import division
import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import OneClassSVM
from sklearn.mixture import GMM

import matplotlib.pyplot as plt
plt.rcParams['figure.autolayout'] = True

from cwc.synthetic_data.datasets import Data
from cwc.models.discriminative_models import MyDecisionTreeClassifier
from cwc.models.background_check import BackgroundCheck

def test_datasets(dataset_names):
    from sklearn.svm import SVC
    from sklearn.cross_validation import StratifiedKFold
    data = Data(dataset_names=dataset_names)

    def separate_sets(x, y, test_fold_id, test_folds):
        x_test = x[test_folds == test_fold_id, :]
        y_test = y[test_folds == test_fold_id]

        x_train = x[test_folds != test_fold_id, :]
        y_train = y[test_folds != test_fold_id]
        return [x_train, y_train, x_test, y_test]

    n_folds = 2
    accuracies = {}
    for name, dataset in data.datasets.iteritems():
        dataset.print_summary()
        skf = StratifiedKFold(dataset.target, n_folds=n_folds, shuffle=True)
        test_folds = skf.test_folds
        accuracies[name] = np.zeros(n_folds)
        for test_fold in np.arange(n_folds):
            x_train, y_train, x_test, y_test = separate_sets(
                    dataset.data, dataset.target, test_fold, test_folds)

            svc = SVC(C=1.0, kernel='rbf', degree=1, tol=0.01)
            svc.fit(x_train, y_train)
            prediction = svc.predict(x_test)
            accuracies[name][test_fold] = 100*np.mean((prediction == y_test))
            print("Acc = {0:.2f}%".format(accuracies[name][test_fold]))
    return accuracies


def main():
    dataset_names = ['autos', 'car', 'cleveland', 'dermatology', 'ecoli',
                     'flare', 'glass', 'led7digit', 'lymphography', 'nursery',
                     'page-blocks', 'pendigits', 'satimage', 'segment', 'shuttle',
                     'vehicle', 'vowel', 'yeast', 'zoo', 'auslan']

    not_available_yet = ['satimage', 'nursery', 'lymphography', 'auslan',
                         'led7digit', 'yeast']

    valid_dataset_names = [name for name in dataset_names if name not in not_available_yet]

    accuracies = test_datasets(valid_dataset_names)
    for name in valid_dataset_names:
        print("{}. {} Acc = {:.2f}% +- {:.2f}".format(
                np.where(np.array(dataset_names) == name)[0]+1,
                name, accuracies[name].mean(), accuracies[name].std()))

if __name__=='__main__':
    main()
