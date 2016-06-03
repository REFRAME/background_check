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

def separate_sets(x, y, test_fold_id, test_folds):
    x_test = x[test_folds == test_fold_id, :]
    y_test = y[test_folds == test_fold_id]

    x_train = x[test_folds != test_fold_id, :]
    y_train = y[test_folds != test_fold_id]
    return [x_train, y_train, x_test, y_test]


def get_abstention_accuracy(y, predictions):
    n_classes = np.alen(np.unique(y))
    n_correct = np.sum(predictions == y)
    n_predicted = np.sum(predictions != n_classes)
    n_abstained = np.sum(predictions == n_classes)
    accuracy = n_correct / n_predicted
    abstention = n_abstained / np.alen(predictions)
    return abstention, accuracy


def accuracy_abstention_curve_ferri(y, posteriors, n_ws=11):
    n_classes = np.alen(np.unique(y))
    ks = np.ones(n_classes) / n_classes
    ws = np.linspace(0.0, 1.01, n_ws)
    accuracies = np.zeros(n_ws)
    abstentions = np.zeros(n_ws)
    for index, w in enumerate(ws):
        taus = (1.0 - ks) * w + ks
        predictions = np.argmax(posteriors / taus, axis=1)
        predictions[posteriors[np.arange(np.alen(predictions)),
                               predictions] < taus[predictions]] = n_classes
        abstention, accuracy = get_abstention_accuracy(y, predictions)
        abstentions[index] = abstention
        accuracies[index] = accuracy
    return abstentions, accuracies


def accuracy_abstention_curve_bc(y, posteriors, x_train, x_test, n_mus=11):
    mus = np.linspace(0.0, 1.01, n_mus)
    accuracies = np.zeros(n_mus)
    abstentions = np.zeros(n_mus)
    #sv = OneClassSVM(nu=0.1, gamma=0.5)
    bc = BackgroundCheck()
    bc.fit(x_train)
    for index, mu in enumerate(mus):
        bc_posteriors = bc.predict_proba(x_test, mu=mu, m=0.0)
        p = np.hstack((posteriors*bc_posteriors[:, 1].reshape(-1, 1),
                       bc_posteriors[:, 0].reshape(-1, 1)))
        predictions = np.argmax(p, axis=1)
        abstention, accuracy = get_abstention_accuracy(y, predictions)
        abstentions[index] = abstention
        accuracies[index] = accuracy
    return abstentions, accuracies


def prune_curve(abst, acc):
    n = np.alen(abst)
    for i in np.arange(n-1, 0, -1):
        if acc[i] == acc[i-1]:
            abst[i] = -1
            acc[i] = -1
    return abst[acc > -1], acc[acc > -1]


if __name__ == '__main__':
    dataset_names = ['tic-tac', 'spambase', 'diabetes']
    data = Data(dataset_names=dataset_names)
    plt.ion()
    np.random.seed(42)
    mc_iterations = 20
    n_folds = 5
    n_ws = 100
    for i, (name, dataset) in enumerate(data.datasets.iteritems()):
        accuracies_ferri = np.zeros((mc_iterations * n_folds, n_ws))
        abstentions_ferri = np.zeros((mc_iterations * n_folds, n_ws))
        accuracies_bc = np.zeros((mc_iterations * n_folds, n_ws))
        abstentions_bc = np.zeros((mc_iterations * n_folds, n_ws))
        data.sumarize_datasets(name)
        for mc in np.arange(mc_iterations):
            skf = StratifiedKFold(dataset.target, n_folds=n_folds,
                                  shuffle=True)
            test_folds = skf.test_folds
            for test_fold in np.arange(n_folds):
                x_train, y_train, x_test, y_test = separate_sets(
                        dataset.data, dataset.target, test_fold, test_folds)


                tree = MyDecisionTreeClassifier(min_samples_leaf=2)
                tree.fit(x_train, y_train)
                posteriors = tree.predict_proba(x_test)

                abst, accs = accuracy_abstention_curve_ferri(y_test,
                                                             posteriors,
                                                             n_ws=n_ws)

                accuracies_ferri[mc * n_folds + test_fold] = accs
                abstentions_ferri[mc * n_folds + test_fold] = abst

                abst_bc, accs_bc = accuracy_abstention_curve_bc(y_test,
                                                          posteriors,
                                                          x_train, x_test,
                                                          n_mus=n_ws)

                accuracies_bc[mc * n_folds + test_fold] = accs_bc
                abstentions_bc[mc * n_folds + test_fold] = abst_bc

        # roc = roc_curve(y_test, posteriors, pos_label=0)
        mean_abst_ferri, mean_acc_ferri = prune_curve(
            abstentions_ferri.mean(axis=0), accuracies_ferri.mean(axis=0))
        mean_eff_ferri = (mean_acc_ferri - mean_abst_ferri + 1.0) / 2.0
        mean_eff_ferri = mean_eff_ferri[np.logical_not(
            np.isnan(mean_eff_ferri))].mean()

        mean_abst_bc, mean_acc_bc = prune_curve(
            abstentions_bc.mean(axis=0), accuracies_bc.mean(axis=0))
        mean_eff_bc = (mean_acc_bc - mean_abst_bc + 1.0) / 2.0
        mean_eff_bc = mean_eff_bc[np.logical_not(np.isnan(mean_eff_bc))].mean()

        fig = plt.figure("Background Check {}".format(name))
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(mean_abst_ferri, mean_acc_ferri, 'b.-',
                label="Ferri (mean efficacy = {0:.2f})".format(mean_eff_ferri))
        ax.plot(mean_abst_bc, mean_acc_bc, 'r.--',
                label="BC (mean efficacy = {0:.2f})".format(mean_eff_bc))
        ax.set_xlabel('Abstention')
        ax.set_ylabel('Accuracy')
        ax.legend(loc='lower right')
        ax.set_title(name)

        plt.show()
