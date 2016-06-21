from __future__ import division
import os
from optparse import OptionParser
import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC

from sklearn.svm import OneClassSVM
from sklearn.mixture import GMM
from sklearn.neighbors import KernelDensity

from cwc.data_wrappers.datasets import Data
from cwc.models.discriminative_models import MyDecisionTreeClassifier
from cwc.models.background_check import BackgroundCheck
from cwc.models.oc_decomposition import OcDecomposition
from cwc.models.density_estimators import MyMultivariateKernelDensity

import pandas as pd
from diary import Diary
import copy

# Not to crop the output columns
pd.set_option('expand_frame_repr', False)


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


def export_datasets_description_to_latex(data, path='', index=True):
    df_data = MyDataFrame(columns=['Name', 'Samples', 'Features', 'Classes'])
    dataset_names = data.datasets.keys()
    dataset_names.sort()
    for name in dataset_names:
        dataset = data.datasets[name]
        df_data = df_data.append_rows([[name, dataset.data.shape[0],
                             dataset.data.shape[1], len(dataset.classes)]])

    def float_to_int_string(x):
        return '%1.0f' % x

    df_data.index += 1
    df_data.to_latex(os.path.join(path,'datasets.tex'),
                     float_format=float_to_int_string , index=index)


def export_summary(df, diary):
    def float_100_to_string(x):
        return '%2.2f' % (100*x)

    df = df.convert_objects(convert_numeric=True)
    table = df.pivot_table(values=['acc', 'logloss'], index=['dataset'],
                                  columns=['method'], aggfunc=[np.mean])

    table.to_latex(os.path.join(diary.path,'acc.tex'),
                   float_format=float_100_to_string)

    table = df.pivot_table(values=['acc', 'logloss'], index=['dataset'],
                                  columns=['method'], aggfunc=[np.mean,
                                      np.std])

    table.to_latex(os.path.join(diary.path,'acc_std.tex'),
                   float_format=float_100_to_string)

    diary.add_entry('summary', [table])


def fit_estimators(base_estimator, X, y):
    estimators = []
    bcs = []
    classes = np.unique(y)
    n_classes = np.alen(classes)
    for c_index in np.arange(n_classes):
        c = copy.deepcopy(base_estimator)
        c.fit(X[y == c_index])
        estimators.append(c)
        bc = BackgroundCheck(estimator=base_estimator)
        bc.set_estimator(c, X[y == c_index])
        bcs.append(bc)
    return estimators, bcs


def main(dataset_names=None, estimator_type="kernel", mc_iterations=1,
        n_folds=10, seed_num=42):
    if dataset_names is None:
        dataset_names = ['glass', 'hepatitis', 'ionosphere', 'vowel']

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

    bandwidth_o_norm = 0.05
    bandwidth_t_norm = 0.05
    bandwidth_bc = 0.05

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
                                   'estimator_type', estimator_type,
                                   'bw_o', bandwidth_o_norm,
                                   'bw_t', bandwidth_t_norm,
                                   'bw_bc', bandwidth_bc])
    data = Data(dataset_names=dataset_names)
    for name, dataset in data.datasets.iteritems():
        if name in ['letter','shuttle']:
            dataset.reduce_number_instances(0.1)
    export_datasets_description_to_latex(data, path=diary.path)

    for i, (name, dataset) in enumerate(data.datasets.iteritems()):
        np.random.seed(seed_num)
        dataset.print_summary()
        diary.add_entry('datasets', [dataset.__str__()])
        # accuracies_tuned = np.zeros(mc_iterations * n_folds)
        # if name in bandwidths_o_norm.keys():
        #     bandwidth_o_norm = bandwidths_o_norm[name]
        #     bandwidth_t_norm = bandwidths_t_norm[name]
        #     bandwidth_bc = bandwidths_bc[name]
        # else:
        #     bandwidth_o_norm = np.mean(bandwidths_o_norm.values())
        #     bandwidth_t_norm = np.mean(bandwidths_t_norm.values())
        #     bandwidth_bc = np.mean(bandwidths_bc.values())
        for mc in np.arange(mc_iterations):
            skf = StratifiedKFold(dataset.target, n_folds=n_folds,
                                  shuffle=True)
            test_folds = skf.test_folds
            for test_fold in np.arange(n_folds):
                x_train, y_train, x_test, y_test = separate_sets(
                        dataset.data, dataset.target, test_fold, test_folds)

                # if name in ['glass', 'hepatitis', 'ionosphere', 'thyroid',
                #             'iris', 'heart-statlog', 'diabetes', 'abalone',
                #             'mushroom', 'spambase']:
                x_test, y_test = generate_outliers(x_test, y_test)
                # elif name == 'vowel':
                #     x_train = x_train[y_train <= 5]
                #     y_train = y_train[y_train <= 5]
                #     y_test[y_test > 5] = 6
                # elif dataset.n_classes > 2:
                #     x_train = x_train[y_train <= dataset.n_classes/2]
                #     y_train = y_train[y_train <= dataset.n_classes/2]
                #     y_test[y_test > dataset.n_classes/2] = dataset.n_classes+1
                # else:
                #     continue

                if estimator_type == "svm":
                    est = OneClassSVM(nu=0.5, gamma=1.0/x_train.shape[1])
                elif estimator_type == "gmm":
                    est = GMM(n_components=1)
                elif estimator_type == "gmm3":
                    est = GMM(n_components=3)
                elif estimator_type == "kernel":
                    est = MyMultivariateKernelDensity(kernel='gaussian',
                                                      bandwidth=bandwidth_bc)
                estimators = None
                bcs = None
                if estimator_type == "kernel":
                    estimators, bcs = fit_estimators(
                                                  MyMultivariateKernelDensity(
                                                      kernel='gaussian',
                                                      bandwidth=bandwidth_bc),
                                        x_train, y_train)

                # Untuned background check
                bc = BackgroundCheck(estimator=est, mu=0.0, m=1.0)
                oc = OcDecomposition(base_estimator=bc)
                if estimators is None:
                    oc.fit(x_train, y_train)
                else:
                    oc.set_estimators(bcs, x_train, y_train)
                accuracy = oc.accuracy(x_test, y_test)
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
                if estimators is None:
                    oc_o_norm.fit(x_train, y_train)
                else:
                    oc_o_norm.set_estimators(estimators, x_train, y_train)
                accuracy_o_norm = oc_o_norm.accuracy(x_test, y_test)
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
                if estimators is None:
                    oc_t_norm.fit(x_train, y_train)
                else:
                    oc_t_norm.set_estimators(estimators, x_train, y_train)
                accuracy_t_norm = oc_t_norm.accuracy(x_test, y_test)
                diary.add_entry('validation', ['dataset', name,
                                               'method', 'T-norm',
                                               'mc', mc,
                                               'test_fold', test_fold,
                                               'acc', accuracy_t_norm])
                df = df.append_rows([[name, 'T-norm', mc, test_fold,
                                      accuracy_t_norm]])

                # Tuned background check
                # if name in tuned_mus.keys():
                #     mus = tuned_mus[name]
                #     ms = tuned_ms[name]
                # else:
                #     mus = None
                #     ms = None
                # bc = BackgroundCheck(estimator=est, mu=0.0, m=1.0)
                # oc_tuned = OcDecomposition(base_estimator=bc)
                # oc_tuned.fit(x_train, y_train, mus=mus, ms=ms)
                # accuracy_tuned = oc_tuned.accuracy(x_test, y_test, mus=mus,
                #                                    ms=ms)
                # accuracies_tuned[mc * n_folds + test_fold] = accuracy_tuned
                # diary.add_entry('validation', ['dataset', name,
                #                                'method', 'BC-tuned',
                #                                'mc', mc,
                #                                'test_fold', test_fold,
                #                                'acc', accuracy_tuned])
                # df = df.append_rows([[name, 'BC-tuned', mc, test_fold,
                #                       accuracy_tuned]])
    export_summary(df, diary)

def parse_arguments():
    parser = OptionParser()
    parser.add_option("-d", "--dataset-names", dest="dataset_names",
            default=None, help="list of dataset names coma separated")
    parser.add_option("-e", "--estimator", dest="estimator_type",
            default='kernel', type='string',
            help="Estimator to use for the background check")
    parser.add_option("-m", "--mc-iterations", dest="mc_iterations",
            default=20, type=int,
            help="Number of Monte Carlo iterations")
    parser.add_option("-f", "--n-folds", dest="n_folds",
            default=5, type=int,
            help="Number of folds for the cross-validation")
    parser.add_option("-s", "--seed-num", dest="seed_num",
            default=42, type=int,
            help="Seed number for the random number generator")

    return parser.parse_args()

if __name__ == '__main__':
    (options, args) = parse_arguments()

    if options.dataset_names is not None:
        dataset_names = options.dataset_names.split(',')
    else:
        dataset_names = None

    main(dataset_names, options.estimator_type, options.mc_iterations,
            options.n_folds, options.seed_num)
