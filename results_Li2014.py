from __future__ import division
import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import OneClassSVM
from sklearn.mixture import GMM
from sklearn.svm import SVC
import matplotlib.pyplot as plt
plt.rcParams['figure.autolayout'] = True

from cwc.synthetic_data.datasets import Data
from cwc.models.ovo_classifier import OvoClassifier
from cwc.models.confident_classifier import ConfidentClassifier
from cwc.models.ensemble import Ensemble


def separate_sets(x, y, test_fold_id, test_folds):
    x_test = x[test_folds == test_fold_id, :]
    y_test = y[test_folds == test_fold_id]

    x_train = x[test_folds != test_fold_id, :]
    y_train = y[test_folds != test_fold_id]
    return [x_train, y_train, x_test, y_test]


def main():
    # All the datasets used in Li2014
    #dataset_names = ['abalone', 'balance-scale', 'credit-approval',
    #'dermatology', 'ecoli', 'german', 'heart-statlog', 'hepatitis', 'horse',
    #'ionosphere', 'lung-cancer', 'libras-movement', 'mushroom', 'diabetes',
    #'landsat-satellite', 'segment', 'spambase', 'breast-cancer-w', 'yeast']

    dataset_names = ['heart-statlog']
    data = Data(dataset_names=dataset_names)
    np.random.seed(42)
    mc_iterations = 20
    n_folds = 5
    n_ensemble = 20
    estimator_type = "gmm"
    print estimator_type
    for i, (name, dataset) in enumerate(data.datasets.iteritems()):
        dataset.print_summary()
        accuracies = np.zeros(mc_iterations * n_folds)
        accuracies_li = np.zeros(mc_iterations * n_folds)
        for mc in np.arange(mc_iterations):
            skf = StratifiedKFold(dataset.target, n_folds=n_folds,
                                  shuffle=True)
            test_folds = skf.test_folds
            for test_fold in np.arange(n_folds):
                x_train, y_train, x_test, y_test = separate_sets(
                        dataset.data, dataset.target, test_fold, test_folds)

                sv = SVC(kernel='linear', probability=True)
                if estimator_type == "svm":
                    est = OneClassSVM(nu=0.1, gamma='auto')
                elif estimator_type == "gmm":
                    est = GMM(n_components=1)
                classifier = ConfidentClassifier(classifier=sv,
                                                 estimator=est, mu=0.5,
                                                 m=0.5)
                ovo = OvoClassifier(base_classifier=classifier)
                ensemble = Ensemble(base_classifier=ovo,
                                    n_ensemble=n_ensemble)
                ensemble.fit(x_train, y_train)
                accuracy = ensemble.accuracy(x_test, y_test)
                accuracies[mc * n_folds + test_fold] = accuracy

                ensemble_li = Ensemble(n_ensemble=n_ensemble, lambd=1e-8)
                ensemble_li.fit(x_train, y_train)
                accuracy_li = ensemble_li.accuracy(x_test, y_test)
                accuracies_li[mc * n_folds + test_fold] = accuracy_li

    print('Mean accuracy BC={}'.format(np.mean(accuracies)))
    print('Std accuracy BC={}'.format(np.std(accuracies)))
    print('Mean accuracy Li={}'.format(np.mean(accuracies_li)))
    print('Std accuracy Li={}'.format(np.std(accuracies_li)))


if __name__ == '__main__':
    main()
