from __future__ import division
import numpy as np
np.random.seed(42)
from tensorflow.examples.tutorials.mnist import input_data

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.ion()
plt.rcParams['figure.figsize'] = (7,4)
plt.rcParams['figure.autolayout'] = True

from sklearn.mixture import GMM
from sklearn import svm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

from cwc.synthetic_data import toy_examples
from cwc.synthetic_data import reject
from cwc.evaluation.rgrpg import RGRPG
from cwc.evaluation.rgp import RGP
from cwc.evaluation.abstaingaincurve import AbstainGainCurve
from cwc.evaluation.metrics import average_cross_entropy
from cwc.evaluation.metrics import composite_average_cross_entropy
from cwc.evaluation.metrics import compute_cace
from cwc.evaluation.metrics import calculate_mo_ce
from cwc.visualization import heatmaps
from cwc.visualization import scatterplots

from diary import Diary

def train_reject_model(x, r):
    """Train a classifier of training points

    Returns a classifier that predicts high probability values for training
    points and low probability values for reject points.
    """
    model_rej = svm.SVC(probability=True)
    #model_rej = tree.DecisionTreeClassifier(max_depth=3)

    xr = np.vstack((x,r))
    y = np.hstack((np.ones(np.alen(x)), np.zeros(np.alen(r)))).T
    model_rej.fit(xr, y)

    return model_rej


def train_classifier_model(x,y):
    model_clas = svm.SVC(probability=True)
    #model_clas = tree.DecisionTreeClassifier(max_depth=3)
    model_clas = model_clas.fit(x,y)
    return model_clas


if __name__ == "__main__":
    diary = Diary(name='digits_vs_letters', path='results', overwrite=False,
                  fig_format='svg')
    diary.add_notebook('training', verbose=True)
    diary.add_notebook('validation', verbose=True)

    n_thresholds = 100

    #####################################################
    # TRAINING                                          #
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
    y_valid = mnist.validation.labels.argmax(axis=1)
    mnist = None
    indices = np.random.choice(range(len(x_train)), 5000, replace=False)
    x_train = x_train[indices]
    y_train = y_train[indices]
    x_train_y = x_train_y[indices]

    r_train = reject.create_reject_data(x_train, proportion=1,
            method='uniform_hsphere', pca=True, pca_variance=0.99,
            pca_components=0, hshape_cov=0, hshape_prop_in=0.99,
            hshape_multiplier=1.5)

    fig = plt.figure('training_data')
    fig.clf()
    scatterplots.plot_data_and_reject(x_train,y_train,r_train,fig=fig)
    diary.save_figure(fig,'training_data')

    # Classifier of reject data
    model_rej = train_reject_model(x_train, r_train)

    # TRAINING SCORES
    c1_rs = model_rej.predict_proba(r_train)
    c1_ts = model_rej.predict_proba(x_train)

    c2_rs = np.zeros((r_train.shape[0], n_classes))
    c2_ts = x_train_y

    ace1, ace2, cace = compute_cace(c1_rs, c1_ts, c2_ts, y_train)

    print('TRAIN RESULTS')
    diary.add_entry('training', ['Step1 ACE', ace1])
    diary.add_entry('training', ['Step2 ACE', ace2])
    diary.add_entry('training', ['Composite ACE', cace])

    ce_bas, ce_cco, ce_clas, ce_rej, ce_bas_rej = calculate_mo_ce(
                                        c2_ts, c2_rs, c1_ts, c1_rs, y_train)
    diary.add_entry('training', ['CE clas', ce_clas])
    diary.add_entry('training', ['CE rej', ce_rej])
    diary.add_entry('training', ['CE bas_rej', ce_bas_rej])
    diary.add_entry('training', ['CE baseline', ce_bas])
    diary.add_entry('training', ['CE CCO', ce_cco])

    step1_reject_scores = c1_rs[:,1]
    step1_training_scores = c1_ts[:,1]
    step2_training_scores = c2_ts[:,1]
    training_labels = y_train

    step1_acc = (np.sum(step1_reject_scores < 0.5) +
                 np.sum(step1_training_scores >= 0.5)
                )/(np.alen(x_train)+np.alen(r_train))
    step1_prior = (np.max([np.alen(x_train),np.alen(r_train)])
            /(np.alen(x_train)+np.alen(r_train)))
    diary.add_entry('training', ['Step1 ACC', step1_acc])
    diary.add_entry('training', ['Step1 prior', step1_prior])

    step2_acc = np.mean(np.argmax(c2_ts, axis=1) == y_train)
    step2_prior = np.max([1-np.mean(y_train), np.mean(y_train)])
    diary.add_entry('training', ['Step2 ACC', step2_acc])
    diary.add_entry('training', ['Step2 prior', step2_prior])

    #####################################################
    # VALIDATION                                        #
    #####################################################
    x_valid_h = np.load('datasets/mnist_valid_h_fc1_no_relu.npy')
    x_valid_y = np.load('datasets/mnist_valid_y.npy')
    x_valid = np.hstack([x_valid_h, x_valid_y])
    x_valid = scaler.transform(x_valid)
    x_valid_h = None

    r_valid_h = np.load('datasets/chars74k_letters_h_fc1_no_relu.npy')
    r_valid_y = np.load('datasets/chars74k_letters_y.npy')
    r_valid = np.hstack([r_valid_h, r_valid_y])
    r_valid_h = None
    r_valid = scaler.transform(r_valid)


    indices = np.random.choice(range(len(x_valid)), len(r_valid), replace=False)
    x_valid = x_valid[indices]
    y_valid = y_valid[indices]
    x_valid_y = x_valid_y[indices]

    fig = plt.figure('validation_data')
    fig.clf()
    scatterplots.plot_data_and_reject(x_valid,y_valid,r_valid,fig=fig)
    diary.save_figure(fig,'validation_data.pdf')

    # VALIDATION SCORES
    c1_rs = model_rej.predict_proba(r_valid)
    c1_ts = model_rej.predict_proba(x_valid)
    c2_rs = r_valid_y
    c2_ts = x_valid_y

    ace1, ace2, cace = compute_cace(c1_rs, c1_ts, c2_ts, y_valid)

    print('VALIDATION RESULTS')
    diary.add_entry('validation', ['Step1 ACE', ace1])
    diary.add_entry('validation', ['Step2 ACE', ace2])
    diary.add_entry('validation', ['Composite ACE', cace])

    ce_bas, ce_cco, ce_clas, ce_rej, ce_bas_rej = calculate_mo_ce(
                                        c2_ts, c2_rs, c1_ts, c1_rs, y_valid)
    diary.add_entry('validation', ['CE clas', ce_clas])
    diary.add_entry('validation', ['CE rej', ce_rej])
    diary.add_entry('validation', ['CE bas_rej', ce_bas_rej])
    diary.add_entry('validation', ['CE baseline', ce_bas])
    diary.add_entry('validation', ['CE CCO', ce_cco])

    step1_reject_scores = c1_rs[:,1]
    step1_training_scores = c1_ts[:,1]
    step2_reject_scores = c2_rs[:,1]
    step2_training_scores = c2_ts[:,1]
    training_labels = y_valid

    step1_acc = (np.sum(step1_reject_scores < 0.5) +
                 np.sum(step1_training_scores >= 0.5)
                )/(np.alen(x_valid)+np.alen(r_valid))
    step1_prior = (np.max([np.alen(x_valid),np.alen(r_valid)])
            /(np.alen(x_valid)+np.alen(r_valid)))
    diary.add_entry('validation', ['Step1 ACC', step1_acc])
    diary.add_entry('validation', ['Step1 prior', step1_prior])

    step2_acc = np.mean(np.argmax(c2_ts, axis=1) == y_valid)
    step2_prior = np.max([1-np.mean(y_valid), np.mean(y_valid)])
    diary.add_entry('validation', ['Step2 ACC', step2_acc])
    diary.add_entry('validation', ['Step2 prior', step2_prior])





    # Area under the RGP curve
    rgp = RGP(step1_reject_scores, step1_training_scores,
              step2_training_scores, training_labels)

    diary.add_entry('validation', ['AUC-RGP', rgp.calculate_area()])
    fig = plt.figure('RGP_acc')
    fig.clf()
    rgp.plot(fig, accuracy=True)
    diary.save_figure(fig,'rgp_acc')

    fig = plt.figure('RGP_pre')
    fig.clf()
    rgp.plot(fig, precision=True)
    diary.save_figure(fig,'rgp_pre')

    ag = AbstainGainCurve(step1_reject_scores, step1_training_scores,
              step2_training_scores, training_labels)

    diary.add_entry('validation', ['AUC-AG', ag.calculate_area()])
    fig = plt.figure('AG')
    fig.clf()
    ag.plot(fig)
    diary.save_figure(fig,'ag')

    diary.add_entry('validation', ['Clas1 RGP threshold', rgp.get_optimal_step1_threshold()])

    print("RG1_PG1_ACC2")
    fig = plt.figure('RG1_PG1_ACC2')
    fig.clf()
    rgrpg = RGRPG(step1_reject_scores, step1_training_scores, step2_training_scores, training_labels)
    rgrpg.plot_simple_3d(fig)
    diary.save_figure(fig,'rg1_pg1_acc2')

    print("ROC_curve_classifier_1")
    fig = plt.figure('ROC_curve_classifier_1')
    fig.clf()
    p1 = np.vstack([np.zeros((len(step1_reject_scores),1)),
                    np.ones((len(step1_training_scores),1))])
    fpr, tpr, thresholds_1 = roc_curve(p1, np.append(step1_reject_scores,
                                                   step1_training_scores))
    area = auc(fpr, tpr)
    diary.add_entry('validation', ['Clas1 AUC', area])
# FIXME there is a problem with some decision trees
#print('PRGA Classifier 1 = {}'.format(rgrpg.prga))
    plt.plot(fpr, tpr, 'k.-')
    plt.xlabel('$FPr_1$')
    plt.ylabel('$TPr_1$')
    diary.save_figure(fig,'roc_clas1')

    thresholds_1 = thresholds_1[thresholds_1 <= 1][::-1]
    ace1s = np.empty_like(thresholds_1)
    ace2s = np.empty_like(thresholds_1)
    caces = np.empty_like(thresholds_1)
    accuracies_1 = np.empty_like(thresholds_1)
    for i_t1, t1 in enumerate(thresholds_1):
        c1_rs = np.empty_like(step1_reject_scores)
        c1_rs[step1_reject_scores >= t1] = 1
        c1_rs[step1_reject_scores < t1] = 0
        c1_rs = np.vstack([1-c1_rs, c1_rs]).transpose()

        c1_ts = np.empty_like(step1_training_scores)
        c1_ts[step1_training_scores >= t1] = 1
        c1_ts[step1_training_scores < t1] = 0
        c1_ts = np.vstack([1-c1_ts, c1_ts]).transpose()

        ace1, ace2, cace = compute_cace(c1_rs, c1_ts, c2_ts, y_valid)
        ace1s[i_t1] = ace1
        ace2s[i_t1] = ace2
        caces[i_t1] = cace
        accuracies_1[i_t1] = np.mean(np.append(c1_rs[:,0], c1_ts[:,1]))

    fig = plt.figure('logloss')
    plt.clf()
    plt.plot(thresholds_1, ace1s, '.-', label='ACE1S')
    plt.plot(thresholds_1, ace2s, '.-', label='ACE2S')
    plt.plot(thresholds_1, caces, '.-', label='CACES')
    plt.plot(thresholds_1, accuracies_1*3.5, '.-', label='ACC1')
    plt.xlabel('thresholds_1')
    plt.ylabel('log-loss')
    plt.xlim([1,0])
    plt.legend()
    diary.save_figure(fig,'log_loss')

# Reducing the problem to three probabilities encoding:
# Reject, Train*positive, Train*negative
    c1_rs = model_rej.predict_proba(r_valid)
    c1_ts = model_rej.predict_proba(x_valid)
    c2_rs = r_valid_y
    c2_ts = x_valid_y

    step1_reject_scores = c1_rs[:,1]
    step1_training_scores = c1_ts[:,1]
    step2_reject_scores = (c2_rs.T*step1_reject_scores).T
    step2_training_scores = (c2_ts.T*step1_training_scores).T

    # p = [R, +, -]
    # q = [R, T*+, T*-]
    q1 = np.expand_dims(np.vstack([c1_rs, c1_ts])[:,0], axis=1)
    p1 = np.vstack([np.ones((len(c1_rs),1)), np.zeros((len(c1_ts),1))])

    q2 = np.vstack([step2_reject_scores, step2_training_scores])
    # label binarize creates [n_samples, n_classes] for n_classes > 2
    # and [n_samples, 1] for n_classes = 2
    p2 = label_binarize(y_valid, np.unique(y_valid))
    if p2.shape[1] == 1:
        p2 = np.hstack([1-p2, p2])
    p2 = np.vstack([np.zeros((len(c1_rs), p2.shape[1])), p2])

    q = np.hstack([q1, q2])
    p = np.hstack([p1, p2])
    print average_cross_entropy(p,q)

    thresholds_joint = np.linspace(0,1,100)
    accuracies_joint = np.empty((len(thresholds_joint), p.shape[1]))
    for i, threshold in enumerate(thresholds_joint):
        prediction = (q >= threshold)
        #prediction[:,0] = 1-np.sum(prediction[:,1:], axis=1)
        accuracies_joint[i] = np.mean(prediction == p, axis=0)

    fig = plt.figure('combined_accuracy')
    plt.clf()
    plt.plot(thresholds_joint, np.mean(accuracies_joint, axis=1), '.-', label='Model_1+2')
    plt.plot(thresholds_joint, accuracies_joint, '.-')
    n_instances = np.sum(p, axis=0)
    n_classes = p.shape[1]
    for c_id in range(n_classes):
        n_at = n_instances[c_id]
        n_nat = sum(n_instances)-n_at
        c_prop = (n_classes*n_at + n_nat)/(n_classes*sum(n_instances))
        plt.plot([0,1], [c_prop, c_prop], label='prop.{}'.format(c_id))
    plt.xlabel('threshold')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0,1])
    diary.save_figure(fig,'combined_accuracy')

    thresholds_joint = np.linspace(0,1,100)
    averaging = 'micro'
    if averaging == 'micro':
        axis = None
        measure_shape = (len(thresholds_joint), 1)
    elif averaging == 'macro':
        axis = 0
        measure_shape = (len(thresholds_joint), p.shape[1])
    elif averaging == 'instance':
        axis = 1
        measure_shape = (len(thresholds_joint), p.shape[0])
    else:
        raise Exception('Averaging \'{}\' not implemented'.format(averaging))
    tp_joint = np.empty(measure_shape)
    fp_joint = np.empty(measure_shape)
    fn_joint = np.empty(measure_shape)
    for i, threshold in enumerate(thresholds_joint):
        prediction = (q >= threshold)
        tp_joint[i] = np.mean(np.logical_and(prediction, p), axis=axis)
        fp_joint[i] = np.mean(np.logical_and(prediction, 1-p), axis=axis)
        fn_joint[i] = np.mean(np.logical_and(1-prediction, p), axis=axis)

    jaccard = tp_joint/(tp_joint + fp_joint + fn_joint)
    fig = plt.figure('jaccard')
    plt.clf()
    plt.plot(thresholds_joint, np.mean(jaccard, axis=1), '.-', label='Multi-jaccard')
    plt.plot(thresholds_joint, jaccard[:,0], '.-', label='Reject')
    for id_c in range(1,jaccard.shape[1]):
        plt.plot(thresholds_joint, jaccard[:,id_c], '.-', label='Class {}'.format(id_c))
    plt.xlabel('threshold')
    plt.ylabel('Jaccard')
    plt.legend(loc='bottom-left')
    diary.save_figure(fig,'jaccard')

    beta = 1
    fbeta = ((1+beta**2)*tp_joint)/((1+beta**2)*tp_joint + beta**2*fn_joint + fp_joint)
    fig = plt.figure('fbeta')
    plt.clf()
    plt.plot(thresholds_joint, fbeta.shape[1]/np.sum(1/fbeta, axis=1), '.-', label='Multi-fbeta')
    plt.plot(thresholds_joint, fbeta[:,0], '.-', label='Reject')
    for id_c in range(1,fbeta.shape[1]):
        plt.plot(thresholds_joint, fbeta[:,id_c], '.-', label='Class {}'.format(id_c))
    plt.xlabel('threshold')
    plt.ylabel('$F_{}$'.format("{"+str(beta)+"}"))
    plt.legend(loc='bottom-left')
    diary.save_figure(fig,'fbeta')

    precision = tp_joint/(tp_joint + fp_joint)
    recall = tp_joint/(tp_joint + fn_joint)
    fig = plt.figure('precision_recall')
    plt.clf()
    plt.plot(np.mean(recall, axis=1), np.mean(precision, axis=1), '.-',
            label='Multi-pre-rec')
    plt.plot(recall[:,0], precision[:,0], '.-', label='Reject')
    for id_c in range(1,recall.shape[1]):
        plt.plot(recall[:,id_c], precision[:,id_c], '.-', label='Class {}'.format(id_c))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.legend(loc='bottom-left')
    diary.save_figure(fig,'pre_rec')
