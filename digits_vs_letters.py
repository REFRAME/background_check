from __future__ import division
import numpy as np
np.random.seed(42)
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
plt.ion()
plt.rcParams['figure.figsize'] = (7,4)
plt.rcParams['figure.autolayout'] = True

from sklearn.preprocessing import StandardScaler

from cwc.data_wrappers import reject
from cwc.evaluation.metrics import compute_cace
from cwc.evaluation.metrics import calculate_mo_ce
from cwc.visualization import scatterplots
from cwc.models.density_estimators import DensityEstimators

from diary import Diary

def initialize_diary():
    diary = Diary(name='digits_vs_letters', path='results', overwrite=False,
                  fig_format='svg')
    diary.add_notebook('training', verbose=True)
    diary.add_notebook('validation', verbose=True)
    return diary

def get_data():
    #####################################################
    # Get training data
    #####################################################
    x_train_h = np.load('datasets/mnist_train_h_fc1_no_relu.npy')
    x_train_y = np.load('datasets/mnist_train_y.npy')
    n_classes = x_train_y.shape[1]
    x_train = np.hstack([x_train_h, x_train_y])
    x_train_h = None

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    y_train = mnist.train.labels.argmax(axis=1)

    indices = np.random.choice(range(len(x_train)), 12000, replace=False)
    x_train = x_train[indices]
    y_train = y_train[indices]
    x_train_y = x_train_y[indices]

    #####################################################
    # Get validation data
    #####################################################
    x_valid_h = np.load('datasets/mnist_valid_h_fc1_no_relu.npy')
    x_valid_y = np.load('datasets/mnist_valid_y.npy')
    x_valid = np.hstack([x_valid_h, x_valid_y])
    x_valid = scaler.transform(x_valid)
    x_valid_h = None
    y_valid = mnist.validation.labels.argmax(axis=1)
    mnist = None

    r_valid_h = np.load('datasets/chars74k_letters_h_fc1_no_relu.npy')
    r_valid_y = np.load('datasets/chars74k_letters_y.npy')
    r_valid = np.hstack([r_valid_h, r_valid_y])
    r_valid_h = None
    r_valid = scaler.transform(r_valid)


    indices = np.random.choice(range(len(x_valid)), len(r_valid), replace=False)
    x_valid = x_valid[indices]
    y_valid = y_valid[indices]
    x_valid_y = x_valid_y[indices]

    return x_train, x_train_y, y_train, x_valid, x_valid_y, y_valid, r_valid

def main():
    diary = initialize_diary()

    x_train, x_train_y, y_train, x_valid, x_valid_y, y_valid, r_valid = get_data()

    #####################################################
    # TRAIN                                             #
    #####################################################
    de = DensityEstimators()
    de.train(x_train, y_train)
    scores = de.predict_proba(x_train)

    acc_clas = np.equal(np.argmax(scores, axis=1), y_train).mean()
    diary.add_entry('training', ['Confidence clas acc', acc_clas])

    classes = de.classes
    for c in classes:
        fig = plt.figure('training_data_{}'.format(c))
        fig.clf()
        scatterplots.plot_data_and_reject(x_train[y_train==c],
                np.ones(np.sum(y_train==c)), de.unknown[c], fig=fig)
        diary.save_figure(fig,'training_data_{}'.format(c))

    fig = plt.figure('hist_mnist_vs_letters')
    fig.clf()
    x_valid_confidence = de.predict_confidence(x_valid)
    r_valid_confidence = de.predict_confidence(r_valid)
    plt.hist([x_valid_confidence, r_valid_confidence])
    diary.save_figure(fig, 'hist_mnist_vs_letters')

    fig = plt.figure('scatter_mnist_vs_letters')
    fig.clf()
    scatterplots.plot_data_and_reject(x_valid, y_valid, r_valid, fig=fig)
    diary.save_figure(fig, 'scatter_mnist_vs_letters')

    scores_x_valid = de.predict_proba(x_valid)
    scores_synthetic = de.scores_agg_unk
    scores_r_valid = de.predict_proba(r_valid)

    fig = plt.figure('scatter_scores_mnist_vs_letters')
    fig.clf()
    scatterplots.plot_data_and_reject(scores_x_valid, y_valid, scores_r_valid, fig=fig)
    diary.save_figure(fig, 'scatter_scores_mnist_vs_letters')

    fig = plt.figure('scatter_scores_mnist_vs_synthetic')
    fig.clf()
    scatterplots.plot_data_and_reject(scores_x_valid, y_valid, scores_synthetic, fig=fig)
    diary.save_figure(fig, 'scatter_scores_mnist_vs_synthetic')

    from IPython import embed
    embed()




#    # Classifier of reject data
#    model_rej = train_confidence_model(x_train, r_train)
#
#    # TRAINING SCORES
#    c1_rs = model_rej.predict_proba(r_train)
#    c1_ts = model_rej.predict_proba(x_train)
#
#    c2_rs = np.zeros((r_train.shape[0], n_classes))
#    c2_ts = x_train_y
#
#    ace1, ace2, cace = compute_cace(c1_rs, c1_ts, c2_ts, y_train)
#
#    print('TRAIN RESULTS')
#    diary.add_entry('training', ['Step1 ACE', ace1])
#    diary.add_entry('training', ['Step2 ACE', ace2])
#    diary.add_entry('training', ['Composite ACE', cace])
#
#    ce_bas, ce_cco, ce_clas, ce_rej, ce_bas_rej = calculate_mo_ce(
#                                        c2_ts, c2_rs, c1_ts, c1_rs, y_train)
#    diary.add_entry('training', ['CE clas', ce_clas])
#    diary.add_entry('training', ['CE rej', ce_rej])
#    diary.add_entry('training', ['CE bas_rej', ce_bas_rej])
#    diary.add_entry('training', ['CE baseline', ce_bas])
#    diary.add_entry('training', ['CE CCO', ce_cco])
#
#    step1_reject_scores = c1_rs[:,1]
#    step1_training_scores = c1_ts[:,1]
#    step2_training_scores = c2_ts[:,1]
#    training_labels = y_train
#
#    step1_acc = (np.sum(step1_reject_scores < 0.5) +
#                 np.sum(step1_training_scores >= 0.5)
#                )/(np.alen(x_train)+np.alen(r_train))
#    step1_prior = (np.max([np.alen(x_train),np.alen(r_train)])
#            /(np.alen(x_train)+np.alen(r_train)))
#    diary.add_entry('training', ['Step1 ACC', step1_acc])
#    diary.add_entry('training', ['Step1 prior', step1_prior])
#
#    step2_acc = np.mean(np.argmax(c2_ts, axis=1) == y_train)
#    step2_prior = np.max([1-np.mean(y_train), np.mean(y_train)])
#    diary.add_entry('training', ['Step2 ACC', step2_acc])
#    diary.add_entry('training', ['Step2 prior', step2_prior])
#
#    #####################################################
#    # VALIDATION                                        #
#    #####################################################
#
#    scores = de.predict(x_valid)
#
#    print("Accuracy = {}".format(np.equal(np.argmax(scores, axis=1),
#        y_valid).mean()))
#
#    scores_r = de.predict(r_valid)
#
#    print("Confidence on reject data = {}".format(np.mean((scores_r >
#        0.95).sum(axis=1) > 0)))
#
#    fig = plt.figure('validation_data')
#    fig.clf()
#    scatterplots.plot_data_and_reject(x_valid,y_valid,r_valid,fig=fig)
#    diary.save_figure(fig,'validation_data.pdf')
#
#    # VALIDATION SCORES
#    c1_rs = model_rej.predict_proba(r_valid)
#    c1_ts = model_rej.predict_proba(x_valid)
#    c2_rs = r_valid_y
#    c2_ts = x_valid_y
#
#    ace1, ace2, cace = compute_cace(c1_rs, c1_ts, c2_ts, y_valid)
#
#    print('VALIDATION RESULTS')
#    diary.add_entry('validation', ['Step1 ACE', ace1])
#    diary.add_entry('validation', ['Step2 ACE', ace2])
#    diary.add_entry('validation', ['Composite ACE', cace])
#
#    ce_bas, ce_cco, ce_clas, ce_rej, ce_bas_rej = calculate_mo_ce(
#                                        c2_ts, c2_rs, c1_ts, c1_rs, y_valid)
#    diary.add_entry('validation', ['CE clas', ce_clas])
#    diary.add_entry('validation', ['CE rej', ce_rej])
#    diary.add_entry('validation', ['CE bas_rej', ce_bas_rej])
#    diary.add_entry('validation', ['CE baseline', ce_bas])
#    diary.add_entry('validation', ['CE CCO', ce_cco])
#
#    step1_reject_scores = c1_rs[:,1]
#    step1_training_scores = c1_ts[:,1]
#    step2_reject_scores = c2_rs[:,1]
#    step2_training_scores = c2_ts[:,1]
#    training_labels = y_valid
#
#    step1_acc = (np.sum(step1_reject_scores < 0.5) +
#                 np.sum(step1_training_scores >= 0.5)
#                )/(np.alen(x_valid)+np.alen(r_valid))
#    step1_prior = (np.max([np.alen(x_valid),np.alen(r_valid)])
#            /(np.alen(x_valid)+np.alen(r_valid)))
#    diary.add_entry('validation', ['Step1 ACC', step1_acc])
#    diary.add_entry('validation', ['Step1 prior', step1_prior])
#
#    step2_acc = np.mean(np.argmax(c2_ts, axis=1) == y_valid)
#    step2_prior = np.max([1-np.mean(y_valid), np.mean(y_valid)])
#    diary.add_entry('validation', ['Step2 ACC', step2_acc])
#    diary.add_entry('validation', ['Step2 prior', step2_prior])

if __name__ == "__main__":
    main()
