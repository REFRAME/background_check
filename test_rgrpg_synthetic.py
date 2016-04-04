from __future__ import division

import numpy as np
np.random.seed(42)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.ion()
plt.rcParams['figure.figsize'] = (7,6)
plt.rcParams['figure.autolayout'] = True

from sklearn.mixture import GMM
from sklearn import svm
from sklearn import tree
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from cwc.synthetic_data import toy_examples
from cwc.synthetic_data import reject
from cwc.evaluation.rgrpg import RGRPG
from cwc.evaluation.rgp import RGP
from cwc.evaluation.abstaingaincurve import AbstainGainCurve
from cwc.evaluation.kurwa import Kurwa

# FIXME change the holes for optional arguments
def generate_data(example=1):
    if example == 1:
        holes=[2,1]
        samples = [900,         # Class 1
                   500]         # Class 2
        means = [[2,2,2],       # Class 1
                 [2,2,3]]       # Class 2
        covs = [[[1,0,0],       # Class 1
                 [0,2,1],
                 [0,1,1]],
                [
                 [1,1,0],       # Class 2
                 [1,2,0],
                 [0,0,1]]]
    elif example == 2:
        holes=[2,2,1]
        samples = [200,         # Class 1
                   200,         # Class 2
                   200]         # Class 3
        means = [[0,0],       # Class 1
                 [-2,0],       # Class 2
                 [3,4]]       # Class 3
        covs = [[[3,-1],       # Class 1
                 [-1,2]],
                [[3,0],       # Class 2
                 [0,3]],
                [[2,1],       # Class 3
                 [1,2]]]
    elif example == 3:
        holes=[2,1]
        samples = [900,         # Class 1
                   50]         # Class 2
        means = [[0,0],       # Class 1
                 [7,4]]       # Class 2
        covs = [[[3,-1],       # Class 1
                 [-1,2]],
                [[2,1],       # Class 2
                 [1,2]]]
    elif example == 4:
        holes=[4,5]
        samples = [900,         # Class 1
                   800]         # Class 2
        means = [[0,0,0,0,0,0,0,0,0,0],       # Class 1
                 [2,2,2,2,2,2,2,2,2,2]]       # Class 2
        A1 = np.random.rand(10,10)
        A2 = np.random.rand(10,10)
        covs = [np.dot(A1,A1.transpose()),
               np.dot(A1,A1.transpose())]
    elif example == 5:
        holes=[2,2,1]
        samples = [100,         # Class 1
                   100,         # Class 2
                   100]         # Class 3
        means = [[0,0],       # Class 1
                 [-3,-3],       # Class 2
                 [2,0]]       # Class 3
        covs = [[[3,-1],       # Class 1
                 [-1,2]],
                [[3,0],       # Class 2
                 [0,3]],
                [[2,1],       # Class 3
                 [1,2]]]

    elif example == 6:
        holes=[3,0]
        samples = [1000,         # Class 1
                   1000]         # Class 2
        means = [[0,0],       # Class 1
                 [0,0]]       # Class 2
        covs = [[[4,0],       # Class 1
                 [0,4]],
                [[2,0],       # Class 2
                 [0,2]]]
    elif example == 7:
        holes=[0,0]
        samples = [1000,         # Class 1
                   1000]         # Class 2
        means = [[0,0],       # Class 1
                 [1,1]]       # Class 2
        covs = [[[1,0],       # Class 1
                 [0,1]],
                [[1,0],       # Class 2
                 [0,1]]]
    else:
        raise Exception('Example {} does not exist'.format(example))

    x, y = toy_examples.generate_gaussians(means=means, covs=covs,
                                           samples=samples, holes=holes)
    if example==5:
        y[y==2] = 1
    return x, y


def plot_data_and_reject(x, y, r, fig=None):
    # Options for plotting
    colors = ['#7777CC', '#FFFF99', '#99FFFF', '#CCCCCC']    # One color per class + reject
    shapes = ['o', 's', '^', '+']

    if fig == None:
        fig = plt.figure('data reject')
    if x.shape[1] >= 3:
        ax = fig.add_subplot(111, projection='3d')
        for k, c in enumerate(colors[:-1]):
            index = (y == k)
            ax.scatter(x[index,0], x[index,1], x[index,2], marker=shapes[k], c=c)
        ax.scatter(r[:,0], r[:,1], r[:,2], marker=shapes[-1], c=colors[-1])
    elif x.shape[1] == 2:
        ax = fig.add_subplot(111)
        for k, c in enumerate(colors[:-1]):
            index = (y == k)
            ax.scatter(x[index,0], x[index,1], marker=shapes[k], c=c)
        ax.scatter(r[:,0], r[:,1], marker=shapes[-1], c=colors[-1])
    elif x.shape[1] == 1:
        ax = fig.add_subplot(111)
        for k, c in enumerate(colors[:-1]):
            index = (y == k)
            ax.hist(x[index,0])
        ax.hist(r[:,0])


def average_cross_entropy(p,q):
    return - np.mean(p*np.log(np.clip(q, 1e-16, 1.0)))


def composite_average_cross_entropy(p1,q1,p2,q2):
    return average_cross_entropy(p1,q1) + average_cross_entropy(p2,q2)


def compute_cace(c1_rs, c1_ts, c2_ts, y):
    q1 = np.vstack([c1_rs, c1_ts])
    p1 = np.vstack([np.ones((len(c1_rs),1)), np.zeros((len(c1_ts),1))])
    p1 = np.hstack([p1, 1-p1])

    q2 = c2_ts
    p2 = label_binarize(y, np.unique(y))
    ace1 = average_cross_entropy(p1,q1)
    ace2 = average_cross_entropy(p2,q2)
    return ace1, ace2, ace1+ace2


def train_reject_model(x, r):
    """Train a classifier of training points

    Returns a classifier that predicts high probability values for training
    points and low probability values for reject points.
    """
    model_rej = svm.SVC(probability=True)
    #model_rej = tree.DecisionTreeClassifier(max_depth=7)

    xr = np.vstack((x,r))
    y = np.hstack((np.ones(np.alen(x)), np.zeros(np.alen(r)))).T
    model_rej.fit(xr, y)

    return model_rej


def train_classifier_model(x,y):
    model_clas = svm.SVC(probability=True)
    #model_clas = tree.DecisionTreeClassifier(max_depth=2)
    model_clas = model_clas.fit(x,y)
    return model_clas


if __name__ == "__main__":
    # for i in  [6]: #range(1,4):
    n_iterations = 1
    n_thresholds = 100
    accuracies = np.empty((n_iterations, n_thresholds))
    recalls = np.empty((n_iterations, n_thresholds))
    for example in [2,3,5,6,7]:
        print('Runing example = {}'.format(example))
        for iteration in range(n_iterations):

            #####################################################
            # TRAINING                                          #
            #####################################################
            x, y = generate_data(example)
            r = reject.create_reject_data(x, proportion=1, method='uniform_hsphere',
                                          pca=True, pca_variance=0.99, pca_components=0,
                                          hshape_cov=0, hshape_prop_in=0.99,
                                          hshape_multiplier=1.5)

            fig = plt.figure('data_reject')
            fig.clf()
            plot_data_and_reject(x,y,r,fig)
            fig.savefig('{}_data_synthetic_example.pdf'.format(example))

            # Classifier of reject data
            model_rej = train_reject_model(x, r)

            # Classifier of training data
            model_clas = train_classifier_model(x, y)

            # TRAINING SCORES
            c1_rs = model_rej.predict_proba(r)
            c1_ts = model_rej.predict_proba(x)
            c2_ts = model_clas.predict_proba(x)

            ace1, ace2, cace = compute_cace(c1_rs, c1_ts, c2_ts, y)

            print('TRAIN RESULTS')
            print('Step1 Average Cross-entropy = {}'.format(ace1))
            print('Step2 Average Cross-entropy = {}'.format(ace2))
            print('Composite Average Cross-entropy = {}'.format(cace))

            step1_reject_scores = c1_rs[:,1]
            step1_training_scores = c1_ts[:,1]
            step2_training_scores = c2_ts[:,1]
            training_labels = y

            print("Step1 Accuracy = {} (prior = {})".format(
                (np.sum(step1_reject_scores < 0.5) +
                 np.sum(step1_training_scores >= 0.5))/(np.alen(x)+np.alen(r)),
                 np.max([np.alen(x),np.alen(r)])/(np.alen(x)+np.alen(r))))

            print("Step2 Accuracy = {} (prior = {})".format(
                np.mean(np.argmax(c2_ts, axis=1) == y),
                np.max([1-np.mean(y), np.mean(y)])))


            #####################################################
            # VALIDATION                                        #
            #####################################################
            # FIXME the validation data does not have the hole in the same
            # position as the training data
            x, y = generate_data(example)
            r = reject.create_reject_data(x, proportion=1, method='uniform_hsphere',
                                          pca=True, pca_variance=0.99, pca_components=0,
                                          hshape_cov=0, hshape_prop_in=0.99,
                                          hshape_multiplier=1.5)

            # TEST SCORES
            c1_rs = model_rej.predict_proba(r)
            c1_ts = model_rej.predict_proba(x)
            c2_ts = model_clas.predict_proba(x)

            ace1, ace2, cace = compute_cace(c1_rs, c1_ts, c2_ts, y)

            print('TEST RESULTS')
            print('Step1 Average Cross-entropy = {}'.format(ace1))
            print('Step2 Average Cross-entropy = {}'.format(ace2))
            print('Composite Average Cross-entropy = {}'.format(cace))

            step1_reject_scores = c1_rs[:,1]
            step1_training_scores = c1_ts[:,1]
            step2_training_scores = c2_ts[:,1]
            training_labels = y

            # Show scores
            # fig = plt.figure('scores')
            # fig.clf()
            # ax = fig.add_subplot(111)
            # ax.set_title('Scores')
            # ax.hist([step1_reject_scores, step1_training_scores,
            #          step2_training_scores])
            # ax.legend(['step1_reject', 'step1_training', 'step2_class1'],
            #           loc='upper center')

            # Evaluate models
            # Accuracy
            print("Step1 Accuracy = {} (prior = {})".format(
                (np.sum(step1_reject_scores < 0.5) +
                 np.sum(step1_training_scores >= 0.5))/(np.alen(x)+np.alen(r)),
                 np.max([np.alen(x),np.alen(r)])/(np.alen(x)+np.alen(r))))

            print("Step2 Accuracy = {} (prior = {})".format(
                np.mean(np.argmax(c2_ts, axis=1) == y),
                np.max([1-np.mean(y), np.mean(y)])))

            # Volume under the PRG-ROC surface
            # rgrpg = RGRPG(step1_reject_scores, step1_training_scores,
            #               step2_training_scores, training_labels)
            #
            # print("Volume = {}".format(rgrpg.calculate_volume()))
            # fig = plt.figure('rgrpg_2d')
            # fig.clf()
            # rgrpg.plot_rgrpg_2d(fig)
            # fig = plt.figure('rgrpg_3d')
            # fig.clf()
            # rgrpg.plot_rgrpg_3d(n_recalls=50, n_points_roc=50, fig=fig)

            kurwa = Kurwa(step1_reject_scores, step1_training_scores,
                          c2_ts, training_labels)

            accuracies[iteration] = kurwa.accuracies
            recalls[iteration] = kurwa.recalls
            print('Iteration {}'.format(iteration))

        expected_acc = accuracies.mean(axis=0)
        expected_rec = recalls.mean(axis=0)
        fig = plt.figure('Expected_rec_acc')
        fig.clf()
        plt.plot(expected_rec, expected_acc, 'k.-')
        plt.errorbar(expected_rec, expected_acc, yerr=3*np.std(accuracies,
                     axis=0), fmt='k.-')
        plt.xlabel('$E[Recall]$')
        plt.ylabel('$E[Accuracy]$')
        plt.xlim([-0.01,1.01])
        plt.ylim([0,1])
        fig.savefig('{}_exp_rec_acc_synthetic_example.pdf'.format(example))

        # Area under the RGP curve
        rgp = RGP(step1_reject_scores, step1_training_scores,
                  step2_training_scores, training_labels)

        print("Area = {}".format(rgp.calculate_area()))
        fig = plt.figure('RGP_acc')
        fig.clf()
        rgp.plot(fig, accuracy=True)
        fig.savefig('{}_rgp_acc_synthetic_example.pdf'.format(example))

        print("Area = {}".format(rgp.calculate_area()))
        fig = plt.figure('RGP_pre')
        fig.clf()
        rgp.plot(fig, precision=True)
        fig.savefig('{}_rgp_pre_synthetic_example.pdf'.format(example))

        print("Area = {}".format(rgp.calculate_area()))
        fig = plt.figure('RGP')
        fig.clf()
        rgp.plot(fig)
        fig.savefig('{}_rgp_synthetic_example.pdf'.format(example))

        ag = AbstainGainCurve(step1_reject_scores, step1_training_scores,
                  step2_training_scores, training_labels)

        print("Area = {}".format(ag.calculate_area()))
        fig = plt.figure('AG')
        fig.clf()
        ag.plot(fig)
        fig.savefig('{}_ag_synthetic_example.pdf'.format(example))

        print("Optimal threshold for the first classifier = {}".format(rgp.get_optimal_step1_threshold()))

        print("RG1_PG1_ACC2")
        fig = plt.figure('RG1_PG1_ACC2')
        fig.clf()
        rgrpg = RGRPG(step1_reject_scores, step1_training_scores, step2_training_scores, training_labels)
        rgrpg.plot_simple_3d(fig)
        fig.savefig('{}_rg1_pg1_acc2_synthetic_example.pdf'.format(example))

        print("ROC_curve_classifier_1")
        fig = plt.figure('ROC_curve_classifier_1')
        fig.clf()
        p1 = np.vstack([np.zeros((len(step1_reject_scores),1)),
                        np.ones((len(step1_training_scores),1))])
        fpr, tpr, thresholds_1 = roc_curve(p1, np.append(step1_reject_scores,
                                                       step1_training_scores))
        area = auc(fpr, tpr)
        print('AUC Classifier 1 = {}'.format(area))
        print('PRGA Classifier 1 = {}'.format(rgrpg.prga))
        plt.plot(fpr, tpr, 'k.-')
        plt.xlabel('$FPr_1$')
        plt.ylabel('$TPr_1$')
        fig.savefig('{}_roc_clas1_synthetic_example.pdf'.format(example))

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

            ace1, ace2, cace = compute_cace(c1_rs, c1_ts, c2_ts, y)
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
        fig.savefig('{}_log_loss_synthetic_example.pdf'.format(example))

        # Reducing the problem to three probabilities encoding:
        # Reject, Train*positive, Train*negative
        c1_rs = model_rej.predict_proba(r)
        c1_ts = model_rej.predict_proba(x)
        c2_rs = model_clas.predict_proba(r)
        c2_ts = model_clas.predict_proba(x)

        step1_reject_scores = c1_rs[:,1]
        step1_training_scores = c1_ts[:,1]
        step2_reject_scores = (c2_rs.T*step1_reject_scores).T
        step2_training_scores = (c2_ts.T*step1_training_scores).T

        # p = [R, +, -]
        # q = [R, T*+, T*-]
        q1 = np.expand_dims(np.vstack([c1_rs, c1_ts])[:,0], axis=1)
        p1 = np.vstack([np.ones((len(c1_rs),1)), np.zeros((len(c1_ts),1))])

        q2 = np.vstack([step2_reject_scores, step2_training_scores])
        p2 = label_binarize(y, np.unique(y))
        # label binarize creates [n_samples, n_classes] for n_classes > 2
        # and [n_samples, 1] for n_classes = 2
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
        fig.savefig('{}_combined_accuracy_synthetic_example.pdf'.format(example))

        thresholds_joint = np.linspace(0,1,100)
        tp_joint = np.empty((len(thresholds_joint), p.shape[1]))
        fp_joint = np.empty((len(thresholds_joint), p.shape[1]))
        fn_joint = np.empty((len(thresholds_joint), p.shape[1]))
        for i, threshold in enumerate(thresholds_joint):
            prediction = (q >= threshold)
            tp_joint[i] = np.mean(np.logical_and(prediction, p), axis=0)
            fp_joint[i] = np.mean(np.logical_and(prediction, 1-p), axis=0)
            fn_joint[i] = np.mean(np.logical_and(1-prediction, p), axis=0)

        jaccard = tp_joint/(tp_joint + fp_joint + fn_joint)
        fig = plt.figure('jaccard')
        plt.clf()
        plt.plot(thresholds_joint, np.mean(jaccard, axis=1), '.-')
        plt.plot(thresholds_joint, jaccard, '.-')
        plt.xlabel('threshold')
        plt.ylabel('Jaccard')
        fig.savefig('{}_jaccard_synthetic_example.pdf'.format(example))

        beta = 1
        fbeta = ((1+beta**2)*tp_joint)/((1+beta**2)*tp_joint + beta**2*fn_joint + fp_joint)
        fig = plt.figure('fbeta')
        plt.clf()
        plt.plot(thresholds_joint, np.mean(fbeta, axis=1), '.-')
        plt.plot(thresholds_joint, fbeta, '.-')
        plt.xlabel('threshold')
        plt.ylabel('fbeta')
        fig.savefig('{}_fbeta_synthetic_example.pdf'.format(example))

        precision = tp_joint/(tp_joint + fp_joint)
        recall = tp_joint/(tp_joint + fn_joint)
        fig = plt.figure('precision_recall')
        plt.clf()
        plt.plot(np.mean(recall, axis=1), np.mean(precision, axis=1), '.-',
                label='Multi-pre-rec')
        for id_c in range(recall.shape[1]):
            plt.plot(recall[:,id_c], precision[:,id_c], '.-', label='Class {}'.format(id_c))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0,1])
        plt.xlim([0,1])
        plt.legend(loc='bottom-left')
        fig.savefig('{}_pre_rec_synthetic_example.pdf'.format(example))


        if x.shape[1] == 2:
            threshold = thresholds_joint[-1-np.argmax(np.mean(fbeta, axis=1)[::-1])]
            # FIXME take into account maximum values for training instances
            # TODO Add heat map for all the predicted probabilities
            # gray to white: reject data
            # blue to alpha: class 1
            # yellow to alpha: class 2
            # cyan to alpha: class 3
            x_min = np.min(r,axis=0)
            x_max = np.max(r,axis=0)
            delta = 50
            x1_lin = np.linspace(x_min[0], x_max[0], delta)
            x2_lin = np.linspace(x_min[1], x_max[1], delta)

            MX1, MX2 = np.meshgrid(x1_lin, x2_lin)
            x_grid = np.asarray([MX1.flatten(),MX2.flatten()]).T
            q1_grid =  model_rej.predict_proba(x_grid)
            q2_grid =  model_clas.predict_proba(x_grid)

            q_grid = np.hstack([np.expand_dims(q1_grid[:,0], axis=1), (q2_grid.T*q1_grid[:,1]).T])
            predictions_grid = q_grid >= threshold
            colors = ['#CCCCCC', '#7777CC', '#FFFF99', '#99FFFF']
            shapes = ['s', 'o', 's', '^']
            sizes = [30, 20, 20, 20]
            fig = plt.figure('fbeta_grid')
            plt.clf()
            for class_id in range(predictions_grid.shape[1]):
                # FIXME consider more than three classes
                plt.scatter(x_grid[predictions_grid[:,class_id],0],
                            x_grid[predictions_grid[:,class_id],1],
                            marker=shapes[class_id],
                            c=colors[class_id], s=sizes[class_id])
            plt.xlim([x_min[0], x_max[0]])
            plt.ylim([x_min[1], x_max[1]])
            fig.savefig('{}_prediction_grid_synthetic_example.pdf'.format(example))

            # CONTOUR CLASSIFIER 1
            fig = plt.figure('data_reject')
            delta = 50
            x_min = np.min(r,axis=0)
            x_max = np.max(r,axis=0)
            x1_lin = np.linspace(x_min[0], x_max[0], delta)
            x2_lin = np.linspace(x_min[1], x_max[1], delta)

            MX1, MX2 = np.meshgrid(x1_lin, x2_lin)
            x_grid = np.asarray([MX1.flatten(),MX2.flatten()]).T
            p_grid =  model_rej.predict_proba(x_grid)
            levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            CS = plt.contour(x_grid[:,0].reshape(delta,delta),
                             x_grid[:,1].reshape(delta,delta),
                             p_grid[:,1].reshape(delta,-1), levels, linewidths=3,
                             alpha=0.5)
            plt.clabel(CS, fontsize=15, inline=2)
            fig.savefig('{}_synthetic_example_reject_contour.pdf'.format(example))
