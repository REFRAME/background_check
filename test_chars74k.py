from __future__ import division
import numpy as np

from cwc.data_wrappers import reject
from cwc.evaluation.confidence_intervals import ConfidenceIntervals
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import minmax_scale
from sklearn import svm
from sklearn.datasets import fetch_mldata
from sklearn import datasets

import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize

def train_reject_model(x, r):
    """Train a classifier of training points

    Returns a classifier that predicts high probability values for training
    points and low probability values for reject points.
    """
    model_rej = svm.SVC(C=1.0, gamma=0.016, kernel='rbf',
                        probability=True)
    xr = np.vstack((x, r))
    yr = np.hstack((np.ones(np.alen(x)), np.zeros(np.alen(r)))).T
    model_rej.fit(xr, yr.astype(int))

    return model_rej


def train_classifier_model(x, y):
    model_clas = svm.SVC(C=10.0, gamma=0.002, kernel='rbf',
                         probability=True)
    model_clas = model_clas.fit(x, y)
    return model_clas


def calculate_mo_accuracy(model_clas_k, model_clas_u, model_rej_k,
                          model_rej_u, yk_test, yu_test):
    yku_test = np.hstack((yk_test, yu_test)).reshape(-1, 1)
    pred_clas_k = np.argmax(model_clas_k, axis=1)
    pred_clas_u = np.argmax(model_clas_u, axis=1)
    pred_clas = np.hstack((pred_clas_k, pred_clas_u)).reshape(-1,1)

    y_rej = np.hstack((np.ones(np.alen(yk_test)), np.zeros(np.alen(yu_test)))).reshape(-1,1)
    pred_rej_k = np.argmax(model_rej_k, axis=1)
    pred_rej_u = np.argmax(model_rej_u, axis=1)
    pred_rej = np.hstack((pred_rej_k, pred_rej_u)).reshape(-1,1)

    multi_y = np.hstack((yku_test, y_rej))
    multi_pred = np.hstack((pred_clas, pred_rej))

    multi_pred_baseline = np.hstack((pred_clas, np.ones((np.alen(pred_rej),1))))

    print('Accuracy f = {}'.format(np.mean(yk_test == pred_clas_k)))
    print('Accuracy fku = {}'.format(np.mean(y_rej == pred_rej)))

    multi_correct = multi_pred_baseline == multi_y
    accuracy_baseline = np.sum(multi_correct) / (np.alen(y_rej) * 2)
    print('Accuracy baseline = {}'.format(accuracy_baseline))

    multi_correct = multi_pred == multi_y
    accuracy_cco = np.sum(multi_correct) / (np.alen(y_rej) * 2)
    print('Accuracy CCO = {}'.format(accuracy_cco))
    return accuracy_baseline, accuracy_cco


def calculate_mo_ce(model_clas_k, model_clas_u, model_rej_k,
                          model_rej_u, yk_test, yu_test):
    p_clas_k = label_binarize(yk_test, np.unique(yk_test))
    p_clas_u = np.ones(p_clas_k.shape) / 10
    p_clas = np.vstack((p_clas_k, p_clas_u))
    q_clas = np.vstack((model_clas_k, model_clas_u))
    e_clas = -np.sum(p_clas * np.log(np.clip(q_clas, 1e-16, 1.0)), axis=1)

    p_rej = np.vstack([np.zeros((np.alen(yk_test), 1)), np.ones((np.alen(
                                                                yu_test),
                                                                1))])
    p_rej = np.hstack([p_rej, 1-p_rej])
    q_rej = np.vstack((model_rej_k, model_rej_u))
    e_rej = -np.sum(p_rej * np.log(np.clip(q_rej, 1e-16, 1.0)), axis=1)

    q_bas_rej = np.zeros(q_rej.shape)
    q_bas_rej[:, 1] = 1.0
    e_bas_rej = -np.sum(p_rej * np.log(np.clip(q_bas_rej, 1e-16, 1.0)), axis=1)

    e_bas = np.sum(e_clas + e_bas_rej) / (np.alen(e_clas) * 2.0)
    e_cco = np.sum(e_clas + e_rej) / (np.alen(e_clas) * 2.0)

    print('Cross-entropy baseline = {}'.format(e_bas))
    print('Cross-entropy CCO = {}'.format(e_cco))

    return e_bas, e_cco

# best for f {'kernel': 'rbf', 'C': 10, 'gamma': 0.002}
# best for fku_hypershpere {'kernel': 'rbf', 'C': 1, 'gamma': 0.016}
# best for fku_hypercube {'kernel': 'rbf', 'C': 1, 'gamma': 0.001}
if __name__ == "__main__":
    np.random.seed(1)

    x = minmax_scale(np.load("datasets/chars74data.npy"), copy=False)
    y = np.load("datasets/chars74target.npy")

    digits_x = x[:550, :]
    digits_y = y[:550]

    letters_x = x[550:, :]
    letters_y = y[550:]

    mc_iterations = 100
    accuracies_baseline = np.empty(mc_iterations)
    accuracies_cco = np.empty(mc_iterations)
    entropies_baseline = np.empty(mc_iterations)
    entropies_cco = np.empty(mc_iterations)

    for iteration in np.arange(mc_iterations):
        print('Iteration = {}'.format(iteration+1))
        X_train, xk_test, y_train, yk_test = train_test_split(digits_x, digits_y,
                                                              test_size=0.5,
                                                              random_state=0,
                                                              stratify=digits_y)

        # Xu_train, xu_test, yu_train, yu_test = train_test_split(letters_x,
        #                                                         letters_y,
        #                                                         test_size=0.5,
        #                                                         random_state=0)

        nk_test = np.alen(yk_test)
        nu_test = np.alen(letters_y)
        chosen_indices = np.random.choice(nu_test, nk_test, replace=False)
        xu_test = letters_x[chosen_indices, :]
        yu_test = letters_y[chosen_indices]

        model_clas = train_classifier_model(X_train, y_train)

        u = reject.create_reject_data(X_train,
                                      proportion=1, method='uniform_hsphere',
                                      pca=True, pca_variance=0,
                                      pca_components=10, hshape_cov=0,
                                      hshape_prop_in=0.99, hshape_multiplier=2)

        # u = np.random.uniform(0.0, 1.0, X_train.shape)

        model_clas_ks = model_clas.predict_proba(X_train)
        model_clas_us = model_clas.predict_proba(u)

        p_x = np.hstack([model_clas_ks, X_train])
        p_u = np.hstack([model_clas_us, u])

        # xu = np.vstack((p_x, p_u))
        # yu = np.hstack((np.ones(np.alen(p_x)), np.zeros(np.alen(p_u)))).T
        #
        # tuned_parameters = [{'kernel': ['rbf'], 'gamma': np.linspace(0.001, 0.1,
        #                                                              100),
        #                      'C': [1, 10, 100, 1000]},
        #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        #
        # print("# Tuning hyper-parameters for fku")
        #
        # clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5,
        #                    scoring='accuracy', verbose=1)
        # clf.fit(xu, yu)
        #
        # print(clf.best_params_)

        model_rej = train_reject_model(p_x, p_u)

        model_clas_k = model_clas.predict_proba(xk_test)
        model_clas_u = model_clas.predict_proba(xu_test)

        p_xk_test = np.hstack([model_clas_k, xk_test])
        p_xu_test = np.hstack([model_clas_u, xu_test])

        model_rej_k = model_rej.predict_proba(p_xk_test)
        model_rej_u = model_rej.predict_proba(p_xu_test)

        # Multi-output accuracies
        accuracies_baseline[iteration], accuracies_cco[iteration] = \
            calculate_mo_accuracy(model_clas_k, model_clas_u, model_rej_k,
                                  model_rej_u, yk_test, yu_test)

        # Multi-output cross-entropies
        entropies_baseline[iteration], entropies_cco[iteration] = \
            calculate_mo_ce(model_clas_k, model_clas_u, model_rej_k,
                            model_rej_u, yk_test, yu_test)

    # Confidence intervals for the multi-output accuracy
    values = np.append(accuracies_baseline.reshape(-1, 1),
                       accuracies_cco.reshape(-1, 1), 1)
    intervals = ConfidenceIntervals(values, ['Baseline', 'CCO'],
                                    n_samples=100, alpha=0.05)
    fig = plt.figure('accuracy')
    plt.title('Multi-output Accuracy')
    intervals.plot(fig)

    # Confidence intervals for the multi-output cross-entropy
    values = np.append(entropies_baseline.reshape(-1, 1),
                       entropies_cco.reshape(-1, 1), 1)
    intervals = ConfidenceIntervals(values, ['Baseline', 'CCO'],
                                    n_samples=100, alpha=0.05)
    fig = plt.figure('cross-entropy')
    plt.title('Multi-output Cross-entropy')
    intervals.plot(fig)
