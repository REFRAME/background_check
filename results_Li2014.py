from __future__ import division
import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import OneClassSVM
from sklearn.mixture import GMM
from sklearn.svm import SVC

from cwc.synthetic_data.datasets import Data
from cwc.models.ovo_classifier import OvoClassifier
from cwc.models.confident_classifier import ConfidentClassifier
from cwc.models.ensemble import Ensemble

import pandas as pd
from diary import Diary

import matplotlib.pyplot as plt
plt.rcParams['figure.autolayout'] = True


def separate_sets(x, y, test_fold_id, test_folds):
    x_test = x[test_folds == test_fold_id, :]
    y_test = y[test_folds == test_fold_id]

    x_train = x[test_folds != test_fold_id, :]
    y_train = y[test_folds != test_fold_id]
    return [x_train, y_train, x_test, y_test]


class MyDataFrame(pd.DataFrame):
    def append_rows(self, rows):
        dfaux = pd.DataFrame(rows, columns=self.columns)
        return self.append(dfaux, ignore_index=True)


def main():
    # All the datasets used in Li2014
    dataset_names = ['abalone', 'balance-scale', 'credit-approval',
    'dermatology', 'ecoli', 'german', 'heart-statlog', 'hepatitis', 'horse',
    'ionosphere', 'lung-cancer', 'libras-movement', 'mushroom', 'diabetes',
    'landsat-satellite', 'segment', 'spambase', 'breast-cancer-w', 'yeast']
    seed_num = 42
    mc_iterations = 20
    n_folds = 5
    n_ensemble = 20
    estimator_type = "gmm"

    # Diary to save the partial and final results
    diary = Diary(name='results_Li2014', path='results', overwrite=False,
                  fig_format='svg')
    # Hyperparameters for this experiment (folds, iterations, seed)
    diary.add_notebook('parameters', verbose=True)
    # Summary for each dataset
    diary.add_notebook('datasets', verbose=True)
    # Partial results for validation
    diary.add_notebook('validation', verbose=True)
    # Final results
    diary.add_notebook('summary', verbose=True)

    columns=['dataset', 'method', 'mc', 'test_fold', 'acc']
    df = MyDataFrame(columns=columns)

    np.random.seed(seed_num)
    diary.add_entry('parameters', ['seed', seed_num, 'mc_it', mc_iterations,
                                   'n_folds', n_folds, 'n_ensemble',
                                   n_ensemble,
                                   'estimator_type', estimator_type])
    data = Data(dataset_names=dataset_names)
    for i, (name, dataset) in enumerate(data.datasets.iteritems()):
        dataset.print_summary()
        diary.add_entry('datasets', [dataset])
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
                diary.add_entry('validation', ['dataset', name,
                                               'method', 'our',
                                               'mc', mc,
                                               'test_fold', test_fold,
                                               'acc', accuracy])
                df = df.append_rows([[name, 'our', mc, test_fold, accuracy]])

                ensemble_li = Ensemble(n_ensemble=n_ensemble, lambd=1e-8)
                ensemble_li.fit(x_train, y_train)
                accuracy_li = ensemble_li.accuracy(x_test, y_test)
                accuracies_li[mc * n_folds + test_fold] = accuracy_li
                diary.add_entry('validation', ['dataset', name,
                                               'method', 'Li2014',
                                               'mc', mc,
                                               'test_fold', test_fold,
                                               'acc', accuracy_li])
                df = df.append_rows([[name, 'Li2014', mc, test_fold, accuracy_li]])

    df = df.convert_objects(convert_numeric=True)
    table = df.pivot_table(values=['acc'], index=['dataset'],
                                  columns=['method'], aggfunc=[np.mean, np.std])
    diary.add_entry('summary', [table])
# <<<<<<< HEAD
#
#     print('Mean accuracy BC={}'.format(np.mean(accuracies)))
#     print('Std accuracy BC={}'.format(np.std(accuracies)))
#     print('Mean accuracy Li={}'.format(np.mean(accuracies_li)))
#     print('Std accuracy Li={}'.format(np.std(accuracies_li)))
# =======


if __name__ == '__main__':
    main()
