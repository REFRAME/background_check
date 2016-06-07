from __future__ import division
import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import OneClassSVM
from sklearn.mixture import GMM
from sklearn.svm import SVC

import matplotlib.pyplot as plt
plt.rcParams['figure.autolayout'] = True

from cwc.synthetic_data.datasets import Data
from cwc.models.discriminative_models import MyDecisionTreeClassifier
from cwc.models.confident_classifier import ConfidentClassifier



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


def prune_ensemble(ensemble, x, n_pruned):
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


def main():
    dataset_names = ['heart-statlog']
    data = Data(dataset_names=dataset_names)
    np.random.seed(42)
    mc_iterations = 20
    n_folds = 5
    n_ensemble = 100
    n_pruned = 10
    bootstrap_percent = 0.75
    estimator_type = "gmm"
    print estimator_type
    for i, (name, dataset) in enumerate(data.datasets.iteritems()):
        dataset.print_summary()
        accuracies = np.zeros(mc_iterations * n_folds)
        for mc in np.arange(mc_iterations):
            skf = StratifiedKFold(dataset.target, n_folds=n_folds,
                                  shuffle=True)
            test_folds = skf.test_folds
            for test_fold in np.arange(n_folds):
                x_train, y_train, x_test, y_test = separate_sets(
                        dataset.data, dataset.target, test_fold, test_folds)

                ensemble = []
                for c_index in np.arange(n_ensemble):
                    x, y = bootstrap(x_train, y_train, bootstrap_percent)
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
                ensemble, weights = prune_ensemble(ensemble, x_train, n_pruned)
                predictions = ensemble_predictions(ensemble, weights, x_test)
                accuracy = np.mean(predictions == y_test)
                accuracies[mc * n_folds + test_fold] = accuracy
    print('Mean accuracy={}'.format(np.mean(accuracies)))
    print('Std accuracy={}'.format(np.std(accuracies)))


if __name__ == '__main__':
    main()
