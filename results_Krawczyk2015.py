from __future__ import division
from optparse import OptionParser
import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC

from sklearn.svm import OneClassSVM
from sklearn.mixture import GMM

# import matplotlib.pyplot as plt
# plt.rcParams['figure.autolayout'] = True

from cwc.data_wrappers.datasets import Data
from cwc.models.discriminative_models import MyDecisionTreeClassifier
from cwc.models.background_check import BackgroundCheck
from cwc.models.oc_decomposition import OcDecomposition

import pandas as pd
from diary import Diary


# def test_datasets(dataset_names):
#     data = Data(dataset_names=dataset_names)
#
#     def separate_sets(x, y, test_fold_id, test_folds):
#         x_test = x[test_folds == test_fold_id, :]
#         y_test = y[test_folds == test_fold_id]
#
#         x_train = x[test_folds != test_fold_id, :]
#         y_train = y[test_folds != test_fold_id]
#         return [x_train, y_train, x_test, y_test]
#
#     n_folds = 2
#     accuracies = {}
#     for name, dataset in data.datasets.iteritems():
#         dataset.print_summary()
#         skf = StratifiedKFold(dataset.target, n_folds=n_folds, shuffle=True)
#         test_folds = skf.test_folds
#         accuracies[name] = np.zeros(n_folds)
#         for test_fold in np.arange(n_folds):
#             x_train, y_train, x_test, y_test = separate_sets(
#                     dataset.data, dataset.target, test_fold, test_folds)
#
#             svc = SVC(C=1.0, kernel='rbf', degree=1, tol=0.01)
#             svc.fit(x_train, y_train)
#             prediction = svc.predict(x_test)
#             accuracies[name][test_fold] = 100*np.mean((prediction == y_test))
#             print("Acc = {0:.2f}%".format(accuracies[name][test_fold]))
#     return accuracies


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


def main(dataset_names=None):
    if dataset_names is None:
        dataset_names = ['autos', 'car', 'cleveland', 'dermatology', 'ecoli',
                         'flare', 'glass', 'led7digit', 'lymphography', 'nursery',
                         'page-blocks', 'pendigits', 'satimage', 'segment',
                         #'shuttle',
                         'vehicle', 'vowel', 'yeast', 'zoo', 'auslan']

    seed_num = 42
    mc_iterations = 5
    n_folds = 2
    estimator_type = "svm"

    # Diary to save the partial and final results
    diary = Diary(name='results_Krawczyk2015', path='results',
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
        for mc in np.arange(mc_iterations):
            skf = StratifiedKFold(dataset.target, n_folds=n_folds,
                                  shuffle=True)
            test_folds = skf.test_folds
            for test_fold in np.arange(n_folds):
                x_train, y_train, x_test, y_test = separate_sets(
                        dataset.data, dataset.target, test_fold, test_folds)

                if estimator_type == "svm":
                    est = OneClassSVM(nu=0.5, gamma=0.5)
                elif estimator_type == "gmm":
                    est = GMM(n_components=3)
                bc = BackgroundCheck(estimator=est)
                oc = OcDecomposition(base_estimator=bc)
                oc.fit(x_train, y_train)
                accuracy = oc.accuracy(x_test, y_test)
                accuracies[mc * n_folds + test_fold] = accuracy
                diary.add_entry('validation', ['dataset', name,
                                               'method', 'our',
                                               'mc', mc,
                                               'test_fold', test_fold,
                                               'acc', accuracy])
                df = df.append_rows([[name, 'our', mc, test_fold, accuracy]])
    df = df.convert_objects(convert_numeric=True)
    table = df.pivot_table(values=['acc'], index=['dataset'],
                                  columns=['method'], aggfunc=[np.mean, np.std])
    diary.add_entry('summary', [table])
    # not_available_yet = ['satimage', 'nursery', 'lymphography', 'auslan',
    #                      'led7digit', 'yeast']
    #
    # valid_dataset_names = [name for name in dataset_names if name not in not_available_yet]

    # accuracies = test_datasets(valid_dataset_names)
    # for name in valid_dataset_names:
    #     print("{}. {} Acc = {:.2f}% +- {:.2f}".format(
    #             np.where(np.array(dataset_names) == name)[0]+1,
    #             name, accuracies[name].mean(), accuracies[name].std()))

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--dataset_names", dest="dataset_names",
                              help="list of dataset names")

    (options, args) = parser.parse_args()
    if hasattr(options, 'dataset_names') and options.dataset_names is not None:
        dataset_names = options.dataset_names.split(',')
        main(dataset_names)
    else:
        main()
