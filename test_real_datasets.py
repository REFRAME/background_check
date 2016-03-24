from __future__ import division
import numpy as np

from cwc.synthetic_data import reject
from cwc.evaluation.rgp import RGP
from cwc.evaluation.confidence_intervals import ConfidenceInterval

from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.mixture import GMM
from sklearn import svm
from sklearn import tree

import matplotlib.pyplot as plt


def load_dataset(name="iris"):
    if name == "iris":
        dataset = datasets.load_iris()
    if name == "digits":
        dataset = datasets.load_digits()
    return dataset.data, dataset.target


def remove_reject_class(data, labels, reject_class=0):
    true_data = data[labels != reject_class, :]
    true_labels = labels[labels != reject_class]
    remaining_labels = np.unique(true_labels)
    counts = [np.sum(true_labels == l) for l in remaining_labels]
    positive_label = remaining_labels[np.argmin(counts)]
    true_labels = 1*(true_labels == positive_label)
    return true_data, true_labels


def separate_reject_class(data, labels, reject_class=0):
    data_reject = data[labels == reject_class, :]
    true_data, true_labels = remove_reject_class(data, labels,
                                                 reject_class=reject_class)
    return data_reject, true_data, true_labels


def separate_sets(x, y, test_fold, validation_fold, test_folds):
    test_data = x[test_folds == test_fold, :]
    test_labels = y[test_folds == test_fold]
    validation_data = x[test_folds == validation_fold, :]
    validation_labels = y[test_folds == validation_fold]
    training_folds = np.logical_and(test_folds != test_fold,
                                    test_folds != validation_fold)
    training_data = x[training_folds, :]
    training_labels = y[training_folds]
    return [training_data, training_labels, validation_data, validation_labels,
            test_data, test_labels]


def train_reject_model(x, r):
    """Train a classifier of training points

    Returns a classifier that predicts high probability values for training
    points and low probability values for reject points.
    """
    # model_rej = svm.SVC(probability=True)
    model_rej = tree.DecisionTreeClassifier(max_depth=4)

    xr = np.vstack((x, r))
    y = np.hstack((np.ones(np.alen(x)), np.zeros(np.alen(r)))).T
    model_rej.fit(xr, y)

    return model_rej


def train_classifier_model(x, y):
    # model_clas = svm.SVC(probability=True)
    model_clas = tree.DecisionTreeClassifier(max_depth=2)
    model_clas = model_clas.fit(x, y)
    return model_clas


def evaluate_test(test_data, test_labels, model_rej=None,
                  model_clas=None, step1_threshold=0.5,
                  step2_threshold=0.5, reject_class=0):
    test_reject, true_test_data, true_test_labels = \
            separate_reject_class(training_data, training_labels,
                                  reject_class=reject_class)
    if model_rej is not None:
        step1_reject_scores = model_rej.predict_proba(test_reject)[:,1]
        step1_test_scores = model_rej.predict_proba(true_test_data)[:,1]
    else:
        step1_reject_scores = np.ones(np.alen(test_reject))
        step1_test_scores = np.ones(np.alen(true_test_data))
    step2_test_scores = model_clas.predict_proba(true_test_data)[:,1]

    n_accepted_rejects = np.sum(step1_reject_scores >= step1_threshold)
    accepted_data = step1_test_scores >= step1_threshold
    n_accepted_data = np.sum(accepted_data)
    n_correct_positives = np.sum(np.logical_and(np.logical_and(
        accepted_data, true_test_labels == 1),
        step2_test_scores >= step2_threshold))
    n_correct_negatives = np.sum(np.logical_and(np.logical_and(
        accepted_data, true_test_labels == 0),
        step2_test_scores < step2_threshold))
    precision_1 = n_accepted_data / (n_accepted_data + n_accepted_rejects)
    n = np.alen(true_test_data)
    recall_1 = n_accepted_data / n
    accuracy_2 = (n_correct_positives + n_correct_negatives) / n
    # return [recall_1, precision_1, accuracy_2]
    print("recall_1 = {}, precision_1 = {}, accuracy_2 = {},"
          " accuracy'_2 = {}".format(recall_1, precision_1, accuracy_2,
                                     accuracy_2 * precision_1))

if __name__ == "__main__":
    np.random.seed(1)
    x, y = load_dataset("iris")
    n_folds = 10
    skf = StratifiedKFold(y, n_folds=n_folds)
    reject_class = 0
    recalls_1 = np.zeros(n_folds)
    precisions_1 = np.zeros(n_folds)
    # accuracies of the second classifier when the first classifier is used
    accuracies_2 = np.zeros(n_folds)
    # accuracies of the second classifier without the first classifier
    acc_2 = np.zeros(n_folds)
    for test_fold in np.arange(n_folds):
        validation_fold = test_fold + 1
        if test_fold == (n_folds - 1):
            validation_fold = 0
        [training_data, training_labels, validation_data, validation_labels,
            test_data, test_labels] = separate_sets(x, y, test_fold,
                                                    validation_fold, skf.test_folds)
        '''
            Training and validation sets preparation. The class that will be
            treated as reject is removed from the data. The generated data
            will be used to simulate knowledge about the reject class.
            The reject class is used for testing.

        '''
        true_training_data, true_training_labels = \
            remove_reject_class(training_data, training_labels,
                                reject_class=reject_class)

        r = reject.create_reject_data(true_training_data,
                                      proportion=1, method='uniform_hsphere',
                                      pca=True, pca_variance=0.9,
                                      pca_components=0, hshape_cov=0,
                                      hshape_prop_in=0.99, hshape_multiplier=2)

        true_validation_data, true_validation_labels = \
            remove_reject_class(validation_data, validation_labels,
                                reject_class=reject_class)

        v_r = reject.create_reject_data(true_validation_data,
                                        proportion=1, method='uniform_hsphere',
                                        pca=True, pca_variance=0.9,
                                        pca_components=0, hshape_cov=0,
                                        hshape_prop_in=0.99, hshape_multiplier=2)

        # Classifier of reject data
        model_rej = train_reject_model(true_training_data, r)

        # Classifier of training data
        model_clas = train_classifier_model(true_training_data,
                                            true_training_labels)
        step1_reject_scores = model_rej.predict_proba(v_r)[:,1]
        step1_validation_scores = model_rej.predict_proba(true_validation_data)[:,1]
        step2_validation_scores = model_clas.predict_proba(true_validation_data)[:,1]

        '''
            TODO: find optimal classification threshold for second classifier
        '''

        rgp = RGP(step1_reject_scores, step1_validation_scores,
                  step2_validation_scores, true_validation_labels,
                  step2_threshold=0.5)

        print("Area = {}".format(rgp.calculate_area()))
        fig = plt.figure('RGP')
        fig.clf()
        rgp.plot(fig)
        step1_threshold = rgp.get_optimal_step1_threshold()
        print("Optimal threshold for the first classifier = {}".format(
            step1_threshold))
        evaluate_test(
            test_data, test_labels, model_rej=model_rej,
                  model_clas=model_clas, step1_threshold=step1_threshold,
                      step2_threshold=0.5, reject_class=reject_class)
        evaluate_test(
            test_data, test_labels, model_rej=None,
                  model_clas=model_clas, step1_threshold=0.5,
                      step2_threshold=0.5, reject_class=reject_class)
    # intervals = ConfidenceInterval()
