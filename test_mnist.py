from __future__ import division
import numpy as np

from cwc.data_wrappers import reject
# from cwc.evaluation.confidence_intervals import ConfidenceIntervals
from sklearn.preprocessing import minmax_scale
from sklearn import svm
from sklearn.datasets import fetch_mldata
from sklearn import datasets

import matplotlib.pyplot as plt


def train_reject_model(x, r):
    """Train a classifier of training points

    Returns a classifier that predicts high probability values for training
    points and low probability values for reject points.
    """
    model_rej = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.55, kernel='rbf',
  max_iter=-1, probability=True, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
    xr = np.vstack((x, r))
    yr = np.hstack((np.ones(np.alen(x)), np.zeros(np.alen(r)))).T
    model_rej.fit(xr, yr.astype(int))

    return model_rej


def train_classifier_model(x, y):
    model_clas = svm.SVC(C=64.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.03125, kernel='rbf',
  max_iter=-1, probability=True, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
    model_clas = model_clas.fit(x, y)
    return model_clas


if __name__ == "__main__":
    np.random.seed(1)

    dataset = fetch_mldata('MNIST original')
    x = minmax_scale(dataset.data, copy=False)
    y = dataset.target.astype(int)

    full_training_set = x[:-10000, :]
    full_training_y = y[:-10000]
    test_set = x[-10000:, :]
    test_y = y[-10000:]

    training_set = np.zeros((10000, 784))
    training_y = np.zeros(10000)

    for digit in np.arange(10):
        indices_digit = np.where(full_training_y == digit)[0]
        chosen_indices = np.random.choice(indices_digit, 1000, replace=False)
        training_set[digit*1000:((digit+1)*1000), :] = full_training_set[
                                                    chosen_indices, :]
        training_y[digit*1000:((digit+1)*1000)] = full_training_y[
            chosen_indices]

    # training_set = np.zeros((1500, 64))
    # training_y = np.zeros(1500)
    # test_set = np.zeros((np.alen(y)-1500, 64))
    # test_y = np.zeros(np.alen(y)-1500)
    # test_position = 0
    # for digit in np.arange(10):
    #     indices_digit = np.where(y == digit)[0]
    #     np.random.shuffle(indices_digit)
    #     training_indices = indices_digit[:150]
    #     test_indices = indices_digit[150:]
    #     training_set[digit*150:((digit+1)*150), :] = x[training_indices, :]
    #     training_y[digit*150:((digit+1)*150)] = y[training_indices]
    #     test_set[test_position:test_position+np.alen(test_indices), :] = x[
    #                                                        test_indices, :]
    #     test_y[test_position:test_position+np.alen(test_indices)] = y[
    #         test_indices]
    #     test_position += np.alen(test_indices)

    unknown_classes = np.random.choice(10, 5, replace=False)
    known_classes = np.where(np.logical_not(np.in1d(np.unique(training_y),
                                                    unknown_classes)))[0]
    '''
        Preparing training set. The classes that
        will be treated as unknown is removed from the training data. The
        generated data will be used to simulate knowledge about the unknown
        classes.
        The unknown classes are used for testing.

    '''

    xk_training = training_set[np.logical_not(np.in1d(training_y,
                                              unknown_classes))]
    yk_training = training_y[np.logical_not(np.in1d(training_y,
                                            unknown_classes))].astype(int)

    # Classifier of training data
    model_clas = train_classifier_model(xk_training, yk_training)

    u = reject.create_reject_data(xk_training,
                                  proportion=1, method='uniform_hsphere',
                                  pca=True, pca_variance=0.9,
                                  pca_components=0, hshape_cov=0,
                                  hshape_prop_in=0.99, hshape_multiplier=2)

    model_clas_ks = model_clas.predict_proba(xk_training)
    model_clas_us = model_clas.predict_proba(u)

    p_x = np.hstack([model_clas_ks, xk_training])
    p_u = np.hstack([model_clas_us, u])

    # Classifier of unknown data
    model_rej = train_reject_model(p_x, p_u)

    xk_test = test_set[np.logical_not(np.in1d(test_y, unknown_classes))]
    yk_test = test_y[np.logical_not(np.in1d(test_y, unknown_classes))]

    xu_test = test_set[np.in1d(test_y, unknown_classes)]
    yu_test = test_y[np.in1d(test_y, unknown_classes)]

    model_clas_k = model_clas.predict_proba(xk_test)
    model_clas_u = model_clas.predict_proba(xu_test)

    p_xk_test = np.hstack([model_clas_k, xk_test])
    p_xu_test = np.hstack([model_clas_u, xu_test])

    model_rej_k = model_rej.predict_proba(p_xk_test)
    model_rej_u = model_rej.predict_proba(p_xu_test)

    yku_test = np.hstack((yk_test, yu_test)).reshape(-1,1)
    pred_clas_k = known_classes[np.argmax(model_clas_k, axis=1)]
    pred_clas_u = known_classes[np.argmax(model_clas_u, axis=1)]
    pred_clas = np.hstack((pred_clas_k, pred_clas_u)).reshape(-1,1)

    y_rej = np.hstack((np.ones(np.alen(yk_test)), np.zeros(np.alen(yu_test)))).reshape(-1,1)
    pred_rej_k = np.argmax(model_rej_k, axis=1)
    pred_rej_u = np.argmax(model_rej_u, axis=1)
    pred_rej = np.hstack((pred_rej_k, pred_rej_u)).reshape(-1,1)

    multi_y = np.hstack((yku_test, y_rej))
    multi_pred = np.hstack((pred_clas, pred_rej))

    from IPython import embed
    embed()

    multi_pred_baseline = np.hstack((pred_clas, np.ones((np.alen(pred_rej),1))))

    multi_correct = multi_pred_baseline == multi_y
    accuracy_baseline = np.sum(multi_correct) / (np.alen(y_rej) * 2)
    print('Accuracy baseline = {}'.format(accuracy_baseline))

    multi_correct = multi_pred == multi_y
    accuracy_cco = np.sum(multi_correct) / (np.alen(y_rej) * 2)
    print('Accuracy CCO = {}'.format(accuracy_cco))
