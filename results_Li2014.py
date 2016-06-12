from __future__ import division
import os
from optparse import OptionParser
import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import OneClassSVM
from sklearn.mixture import GMM
from sklearn.svm import SVC

from cwc.synthetic_data.datasets import Data
from cwc.models.ovo_classifier import OvoClassifier
from cwc.models.confident_classifier import ConfidentClassifier
from cwc.models.ensemble import Ensemble
from cwc.models.density_estimators import MyMultivariateNormal

import pandas as pd
from diary import Diary


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


def export_datasets_description_to_latex(data, path='', index=False):
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
    table = df.pivot_table(values=['acc'], index=['dataset'],
                                  columns=['method'], aggfunc=[np.mean])

    table.to_latex(os.path.join(diary.path,'acc.tex'),
                   float_format=float_100_to_string)

    table = df.pivot_table(values=['acc'], index=['dataset'],
                                  columns=['method'], aggfunc=[np.mean,
                                      np.std])

    table.to_latex(os.path.join(diary.path,'acc_std.tex'),
                   float_format=float_100_to_string)

    diary.add_entry('summary', [table])



def main(dataset_names=None, estimator_type="gmm", mc_iterations=20, n_folds=5,
        n_ensemble=100, seed_num=42):
    if dataset_names is None:
        # All the datasets used in Li2014
        datasets_li2014 = ['abalone', 'balance-scale', 'credit-approval',
        'dermatology', 'ecoli', 'german', 'heart-statlog', 'hepatitis', 'horse',
        'ionosphere', 'lung-cancer', 'libras-movement', 'mushroom', 'diabetes',
        'landsat-satellite', 'segment', 'spambase', 'wdbc', 'wpbc', 'yeast']

        datasets_hempstalk2008 = ['diabetes', 'ecoli', 'glass', 'heart-statlog',
                     'ionosphere', 'iris', 'letter', 'mfeat-karhunen',
                     'mfeat-morphological', 'mfeat-zernike', 'optdigits',
                     'pendigits', 'sonar', 'vehicle', 'waveform-5000']

        datasets_others = [ 'diabetes', 'ecoli', 'glass', 'heart-statlog',
                'ionosphere', 'iris', 'letter', 'mfeat-karhunen',
                'mfeat-morphological', 'mfeat-zernike', 'optdigits',
                'pendigits', 'sonar', 'vehicle', 'waveform-5000',
                'scene-classification', 'tic-tac', 'autos', 'car',
                'cleveland', 'dermatology', 'flare', 'page-blocks', 'segment',
                'shuttle', 'vowel', 'zoo', 'abalone', 'balance-scale',
                'credit-approval', 'german', 'hepatitis', 'lung-cancer']

        # Datasets that we can add but need to be reduced
        datasets_to_add = ['MNIST']

        dataset_names = list(set(datasets_li2014 + datasets_hempstalk2008 +
            datasets_others))

    # Diary to save the partial and final results
    diary = Diary(name='results_Li2014', path='results', overwrite=False,
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
                                   'n_folds', n_folds, 'n_ensemble',
                                   n_ensemble,
                                   'estimator_type', estimator_type])
    data = Data(dataset_names=dataset_names)
    export_datasets_description_to_latex(data, path=diary.path)

    for i, (name, dataset) in enumerate(data.datasets.iteritems()):
        np.random.seed(seed_num)
        dataset.print_summary()
        diary.add_entry('datasets', [dataset.__str__()])
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
                    gamma = 1.0/x_train.shape[1]
                    est = OneClassSVM(nu=0.1, gamma=gamma)
                elif estimator_type == "gmm":
                    est = GMM(n_components=1)
                elif estimator_type == "gmm3":
                    est = GMM(n_components=3)
                elif estimator_type == "mymvn":
                    est = MyMultivariateNormal()
                ovo = OvoClassifier(base_classifier=sv)
                classifier = ConfidentClassifier(classifier=ovo,
                                                 estimator=est, mu=0.5,
                                                 m=0.5)
                ensemble = Ensemble(base_classifier=classifier,
                                    n_ensemble=n_ensemble)
                # classifier = ConfidentClassifier(classifier=sv,
                #                                  estimator=est, mu=0.5,
                #                                  m=0.5)
                # ovo = OvoClassifier(base_classifier=classifier)
                # ensemble = Ensemble(base_classifier=ovo,
                #                     n_ensemble=n_ensemble)
                xs_bootstrap, ys_bootstrap = ensemble.fit(x_train, y_train)
                accuracy = ensemble.accuracy(x_test, y_test)
                accuracies[mc * n_folds + test_fold] = accuracy
                diary.add_entry('validation', ['dataset', name,
                                               'method', 'our',
                                               'mc', mc,
                                               'test_fold', test_fold,
                                               'acc', accuracy])
                df = df.append_rows([[name, 'our', mc, test_fold, accuracy]])

                ensemble_li = Ensemble(n_ensemble=n_ensemble, lambd=1e-8)
                ensemble_li.fit(x_train, y_train, xs=xs_bootstrap,
                                ys=ys_bootstrap)

                accuracy_li = ensemble_li.accuracy(x_test, y_test)
                accuracies_li[mc * n_folds + test_fold] = accuracy_li
                diary.add_entry('validation', ['dataset', name,
                                               'method', 'Li2014',
                                               'mc', mc,
                                               'test_fold', test_fold,
                                               'acc', accuracy_li])
                df = df.append_rows([[name, 'Li2014', mc, test_fold, accuracy_li]])

    export_summary(df, diary)


def parse_arguments():
    parser = OptionParser()
    parser.add_option("-d", "--dataset-names", dest="dataset_names",
            default=None, help="list of dataset names coma separated")
    parser.add_option("-e", "--estimator", dest="estimator_type",
            default='gmm', type='string',
            help="Estimator to use for the background check")
    parser.add_option("-m", "--mc-iterations", dest="mc_iterations",
            default=20, type=int,
            help="Number of Monte Carlo iterations")
    parser.add_option("-f", "--n-folds", dest="n_folds",
            default=5, type=int,
            help="Number of folds for the cross-validation")
    parser.add_option("-n", "--n-ensemble", dest="n_ensemble",
            default=100, type=int,
            help="Number of ensemble models to aggregate")
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
            options.n_folds, options.n_ensemble, options.seed_num)
