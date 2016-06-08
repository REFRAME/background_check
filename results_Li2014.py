from __future__ import division
import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import OneClassSVM
from sklearn.mixture import GMM
from sklearn.svm import SVC
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.rcParams['figure.autolayout'] = True

from cwc.synthetic_data.datasets import Data
from cwc.models.ovo_classifier import OvoClassifier
from cwc.models.confident_classifier import ConfidentClassifier

import pandas as pd
from diary import Diary

def separate_sets(x, y, test_fold_id, test_folds):
    x_test = x[test_folds == test_fold_id, :]
    y_test = y[test_folds == test_fold_id]

    x_train = x[test_folds != test_fold_id, :]
    y_train = y[test_folds != test_fold_id]
    return [x_train, y_train, x_test, y_test]


def bootstrap(x, y, percent):
    n = np.alen(x)
    indices = np.random.choice(n, int(np.around(n * percent)))
    return x[indices], y[indices]


def ensemble_predictions_old(ensemble, x):
    n_ensemble = len(ensemble)
    n = np.alen(x)
    predictions = np.zeros((n, n_ensemble))
    probas = np.zeros((n, n_ensemble))
    confidences = np.zeros((n, n_ensemble))
    for c_index, c in enumerate(ensemble):
        class_proba = c.predict_class_proba(x)
        bc_proba = c.predict_bc_proba(x)
        predictions[:, c_index] = np.argmax(class_proba, axis=1)
        probas[:, c_index] = np.amax(class_proba, axis=1)
        confidences[:, c_index] = bc_proba[:, 1]
    weighted_predictions = predictions * probas * confidences
    weighted_predictions = weighted_predictions.sum(axis=1)
    weighted_predictions /= (probas * confidences).sum(axis=1)
    return np.around(weighted_predictions)


def ensemble_predictions(ensemble, weights, x):
    n = np.alen(x)
    for index, c in enumerate(ensemble):
        class_proba = c.predict_class_proba(x)
        if index == 0:
            predictions = np.zeros((n, class_proba.shape[1]))
        bc_proba = c.predict_bc_proba(x)
        votes = np.argmax(class_proba, axis=1)
        probas = np.amax(class_proba, axis=1)
        predictions[range(n), votes] += probas * weights[index] * bc_proba[:, 1]
    return predictions.argmax(axis=1)


def prune_ensemble_bc(ensemble, x, n_pruned):
    #FIXME the pruning method is wrong
    n_ensemble = len(ensemble)
    confidences = np.zeros(n_ensemble)
    for c_index, c in enumerate(ensemble):
        class_proba = c.predict_class_proba(x)
        bc_proba = c.predict_bc_proba(x)
        confidences[c_index] = np.mean(bc_proba[:, 1])
    confidences = confidences / np.sum(confidences)
    sorted_indices = np.argsort(1.0/(confidences + 1.0))
    pruned_ensemble = [ensemble[sorted_indices[i]] for i in range(n_pruned)]
    return pruned_ensemble, confidences[sorted_indices[:n_pruned]]


def get_weights(ensemble_li, lambd, x, y):
    n_ensemble = len(ensemble_li)
    n = np.alen(y)
    predictions = np.zeros((n, n_ensemble))
    confidences = np.zeros((n, n_ensemble))
    for c_index, c in enumerate(ensemble_li):
        predictions[:, c_index], confidences[:, c_index] = c.predict(x)
    conf_pred = predictions * confidences
    f = lambda w: np.sum(np.power(1.0 - y*(w*conf_pred).sum(axis=1),
                                  2.0)) + lambd*np.linalg.norm(w)
    w0 = np.ones(n_ensemble)/n_ensemble
    cons = ({'type': 'eq', 'fun': lambda w:  1.0 - np.sum(w)})

    bnds = [(0, None) for c_index in range(n_ensemble)]
    res = minimize(f, w0, bounds=bnds, constraints=cons)
    return res.x


def prune_ensemble_li(ensemble_li, x, y, lambd):
    weights = get_weights(ensemble_li, lambd, x, y)
    sorted_indices = np.argsort(1.0/(weights + 1.0))
    n_ensemble = len(ensemble_li)
    n_classes = np.alen(np.unique(y))
    n = np.alen(x)
    accuracies = np.zeros(n_ensemble+1)
    for j in np.arange(n_ensemble+1):
        votes = np.zeros((n, n_classes))
        for c_index in np.arange(0, j):
            i = sorted_indices[c_index]
            pred, conf = ensemble_li[i].predict(x)
            votes[range(n), pred] = votes[range(n), pred] + conf * weights[i]
        if not np.all(votes == 0.0):
            accuracies[j] = np.mean(votes.argmax(axis=1) == y)
    final_j = np.argmax(accuracies)
    pruned_ensemble = [ensemble_li[sorted_indices[i]] for i in range(final_j)]
    return pruned_ensemble, weights[sorted_indices[:final_j]]


def ensemble_predictions_li(ensemble_li, weights, x):
    n = np.alen(x)
    for c_index, c in enumerate(ensemble_li):
        pred, conf = c.predict(x)
        if c_index == 0:
            #TODO create n_classes propoerty in OvoClassifier
            votes = np.zeros((n, c._n_classes))
        votes[range(n), pred] = votes[range(n), pred] + conf * weights[c_index]

    return votes.argmax(axis=1)

class MyDataFrame(pd.DataFrame):
    def append_rows(self, rows):
        dfaux = pd.DataFrame(rows, columns=self.columns)
        return self.append(dfaux, ignore_index=True)

def main():
    # All the datasets used in Li2014
    #dataset_names = ['abalone', 'balance-scale', 'credit-approval',
    #'dermatology', 'ecoli', 'german', 'heart-statlog', 'hepatitis', 'horse',
    #'ionosphere', 'lung-cancer', 'libras-movement', 'mushroom', 'diabetes',
    #'landsat-satellite', 'segment', 'spambase', 'breast-cancer-w', 'yeast']
    dataset_names = ['ionosphere', 'heart-statlog', 'iris']
    seed_num = 42
    mc_iterations = 2 ##20
    n_folds = 2 ##5
    n_ensemble = 10 ##100
    n_pruned = 2 #10
    bootstrap_percent = 0.5 #0.75
    estimator_type = "gmm" #"gmm"

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
                                   n_ensemble, 'n_pruned', n_pruned,
                                   'bootstrap_perc', bootstrap_percent,
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

                ensemble = []
                ensemble_li = []
                for c_index in np.arange(n_ensemble):
                    x, y = bootstrap(x_train, y_train, bootstrap_percent)
                    ovo = OvoClassifier()
                    ovo.fit(x, y)
                    ensemble_li.append(ovo)
                    sv = SVC(kernel='linear', probability=True)
                    if estimator_type == "svm":
                        est = OneClassSVM(nu=0.1, gamma='auto')
                    elif estimator_type == "gmm":
                        est = GMM(n_components=1)
                    classifier = ConfidentClassifier(classifier=sv,
                                                     estimator=est, mu=0.5,
                                                     m=0.5)
                    classifier.fit(x, y)
                    ensemble.append(classifier)

                ensemble_li, weights_li = prune_ensemble_li(ensemble_li,
                                                            x_train, y_train,
                                                            1e-8)

                ensemble, weights = prune_ensemble_bc(ensemble, x_train,
                                                      n_pruned)
                predictions = ensemble_predictions(ensemble, weights, x_test)
                accuracy = np.mean(predictions == y_test)
                accuracies[mc * n_folds + test_fold] = accuracy
                diary.add_entry('validation', ['dataset', name,
                                               'method', 'our',
                                               'mc', mc,
                                               'test_fold', test_fold,
                                               'acc', accuracy])
                df = df.append_rows([[name, 'our', mc, test_fold, accuracy]])


                predictions_li = ensemble_predictions_li(ensemble_li,
                                                         weights_li, x_test)
                accuracy_li = np.mean(predictions_li == y_test)
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


if __name__ == '__main__':
    main()
