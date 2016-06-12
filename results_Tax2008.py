from __future__ import division
from optparse import OptionParser
import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC

from sklearn.svm import OneClassSVM
from sklearn.mixture import GMM
from sklearn.neighbors import KernelDensity

from cwc.synthetic_data.datasets import Data
from cwc.models.discriminative_models import MyDecisionTreeClassifier
from cwc.models.background_check import BackgroundCheck
from cwc.models.oc_decomposition import OcDecomposition
from cwc.models.density_estimators import MyMultivariateKernelDensity

import pandas as pd
from diary import Diary


def separate_sets(x, y, test_fold_id, test_folds):
        x_test = x[test_folds == test_fold_id, :]
        y_test = y[test_folds == test_fold_id]

        x_train = x[test_folds != test_fold_id, :]
        y_train = y[test_folds != test_fold_id]
        return [x_train, y_train, x_test, y_test]


def generate_outliers(x, y, variance_multiplier=4.0, outlier_proportion=0.5):
    outlier_class = np.amax(y) + 1
    n = np.alen(y)
    n_o = int(np.around(outlier_proportion * n))
    means = x.mean(axis=0)
    covs = variance_multiplier*np.cov(x.T)
    outliers = np.random.multivariate_normal(means, covs, n_o)
    x_o = np.vstack((x, outliers))
    y_o = np.append(y, np.ones(n_o) * outlier_class)
    return x_o, y_o


class MyDataFrame(pd.DataFrame):
    def append_rows(self, rows):
        dfaux = pd.DataFrame(rows, columns=self.columns)
        return self.append(dfaux, ignore_index=True)


def main(dataset_names=None):
    if dataset_names is None:
        dataset_names = ['vowel']

    seed_num = 42
    mc_iterations = 1
    n_folds = 10
    estimator_type = "kernel"

    bandwidths_o_norm = {'glass': 0.09, 'hepatitis': 0.105,
                         'ionosphere': 0.039, 'vowel': 0.075}

    bandwidths_bc = {'glass': 0.09, 'hepatitis': 0.105,
                     'ionosphere': 0.039, 'vowel': 0.0145}

    bandwidths_t_norm = {'glass': 0.336, 'hepatitis': 0.015,
                         'ionosphere': 0.0385, 'vowel': 0.0145}

    tuned_mus = {'glass': [0.094, 0.095, 0.2, 0.0, 0.0, 0.1],
                 'vowel': [0.0, 0.0, 0.5, 0.5, 0.5, 0.0]}
    
    tuned_ms = {'glass': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                'vowel': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}

    # Diary to save the partial and final results
    diary = Diary(name='results_Tax2008', path='results',
                  overwrite=False,
                  fig_format='svg')
    # Hyperparameters for this experiment (folds, iterations, seed)
    diary.add_notebook('parameters', verbose=True)
    # Summary for each dataset
    diary.add_notebook('datasets', verbose=False)
    # Partial results for validation
    diary.add_notebook('validation', verbose=True)
    # Final results
    diary.add_notebook('summary', verbose=True)

    columns=['dataset', 'method', 'mc', 'test_fold', 'acc']
    df = MyDataFrame(columns=columns)

    diary.add_entry('parameters', ['seed', seed_num, 'mc_it', mc_iterations,
                                   'n_folds', n_folds,
                                   'estimator_type', estimator_type])
    data = Data(dataset_names=dataset_names)
    for i, (name, dataset) in enumerate(data.datasets.iteritems()):
        np.random.seed(seed_num)
        dataset.print_summary()
        diary.add_entry('datasets', [dataset.__str__()])
        accuracies = np.zeros(mc_iterations * n_folds)
        accuracies_o_norm = np.zeros(mc_iterations * n_folds)
        accuracies_t_norm = np.zeros(mc_iterations * n_folds)
        accuracies_tuned = np.zeros(mc_iterations * n_folds)
        bandwidth_o_norm = bandwidths_o_norm[name]
        bandwidth_t_norm = bandwidths_t_norm[name]
        bandwidth_bc = bandwidths_bc[name]
        for mc in np.arange(mc_iterations):
            skf = StratifiedKFold(dataset.target, n_folds=n_folds,
                                  shuffle=True)
            test_folds = skf.test_folds
            for test_fold in np.arange(n_folds):
                x_train, y_train, x_test, y_test = separate_sets(
                        dataset.data, dataset.target, test_fold, test_folds)

                if name in ['glass', 'hepatitis', 'ionosphere', 'thyroid',
                            'iris', 'heart-statlog', 'diabetes', 'abalone',
                            'mushroom', 'spambase']:
                    x_test, y_test = generate_outliers(x_test, y_test)
                elif name == 'vowel':
                    x_train = x_train[y_train <= 5]
                    y_train = y_train[y_train <= 5]
                    y_test[y_test > 5] = 6

                if estimator_type == "svm":
                    est = OneClassSVM(nu=0.5, gamma=1.0/x_train.shape[1])
                elif estimator_type == "gmm":
                    est = GMM(n_components=1)
                elif estimator_type == "kernel":
                    est = MyMultivariateKernelDensity(kernel='gaussian',
                                                      bandwidth=bandwidth_bc)
                # Untuned background check
                bc = BackgroundCheck(estimator=est, mu=0.0, m=1.0)
                oc = OcDecomposition(base_estimator=bc)
                oc.fit(x_train, y_train)
                accuracy = oc.accuracy(x_test, y_test)
                accuracies[mc * n_folds + test_fold] = accuracy
                diary.add_entry('validation', ['dataset', name,
                                               'method', 'BC',
                                               'mc', mc,
                                               'test_fold', test_fold,
                                               'acc', accuracy])
                df = df.append_rows([[name, 'BC', mc, test_fold, accuracy]])

                e = MyMultivariateKernelDensity(kernel='gaussian',
                                                bandwidth=bandwidth_o_norm)
                oc_o_norm = OcDecomposition(base_estimator=e,
                                            normalization="O-norm")
                oc_o_norm.fit(x_train, y_train)
                accuracy_o_norm = oc_o_norm.accuracy(x_test, y_test)
                accuracies_o_norm[mc * n_folds + test_fold] = accuracy_o_norm
                diary.add_entry('validation', ['dataset', name,
                                               'method', 'O-norm',
                                               'mc', mc,
                                               'test_fold', test_fold,
                                               'acc', accuracy_o_norm])
                df = df.append_rows([[name, 'O-norm', mc, test_fold,
                                      accuracy_o_norm]])

                e = MyMultivariateKernelDensity(kernel='gaussian',
                                                bandwidth=bandwidth_t_norm)
                oc_t_norm = OcDecomposition(base_estimator=e,
                                            normalization="T-norm")
                oc_t_norm.fit(x_train, y_train)
                accuracy_t_norm = oc_t_norm.accuracy(x_test, y_test)
                accuracies_t_norm[mc * n_folds + test_fold] = accuracy_t_norm
                diary.add_entry('validation', ['dataset', name,
                                               'method', 'T-norm',
                                               'mc', mc,
                                               'test_fold', test_fold,
                                               'acc', accuracy_t_norm])
                df = df.append_rows([[name, 'T-norm', mc, test_fold,
                                      accuracy_t_norm]])

                # Tuned background check
                if name in tuned_mus.keys():
                    mus = tuned_mus[name]
                    ms = tuned_ms[name]
                else:
                    mus = None
                    ms = None
                bc = BackgroundCheck(estimator=est, mu=0.0, m=1.0)
                oc_tuned = OcDecomposition(base_estimator=bc)
                oc_tuned.fit(x_train, y_train, mus=mus, ms=ms)
                accuracy_tuned = oc_tuned.accuracy(x_test, y_test, mus=mus,
                                                   ms=ms)
                accuracies_tuned[mc * n_folds + test_fold] = accuracy_tuned
                diary.add_entry('validation', ['dataset', name,
                                               'method', 'BC-tuned',
                                               'mc', mc,
                                               'test_fold', test_fold,
                                               'acc', accuracy_tuned])
                df = df.append_rows([[name, 'BC-tuned', mc, test_fold,
                                      accuracy_tuned]])
    df = df.convert_objects(convert_numeric=True)
    table = df.pivot_table(values=['acc'], index=['dataset'],
                                  columns=['method'], aggfunc=[np.mean, np.std])
    diary.add_entry('summary', [table])


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--dataset_names", dest="dataset_names",
                              help="list of dataset names")

    (options, args) = parser.parse_args()
    if options.dataset_names is not None:
        dataset_names = options.dataset_names.split(',')
        main(dataset_names)
    else:
        main()
