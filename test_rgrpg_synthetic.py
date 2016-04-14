from __future__ import division

import numpy as np
np.random.seed(42)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.ion()
plt.rcParams['figure.figsize'] = (7,4)
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
from cwc.evaluation.metrics import average_cross_entropy
from cwc.evaluation.metrics import composite_average_cross_entropy
from cwc.evaluation.metrics import compute_cace
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
    diary = Diary(name='test_rgrpg', path='results', overwrite=False,
                  fig_format='svg')
    diary.add_notebook('training')
    diary.add_notebook('validation')

    # for i in  [6]: #range(1,4):
    n_iterations = 1
    n_thresholds = 100
    accuracies = np.empty((n_iterations, n_thresholds))
    recalls = np.empty((n_iterations, n_thresholds))
    for example in [2,3,4,5,6,7,8,9]:
        np.random.seed(42)
        print('Runing example = {}'.format(example))
        for iteration in range(n_iterations):

            #####################################################
            # TRAINING                                          #
            #####################################################
            x, y, hole_centers = toy_examples.generate_example(example)
            r = reject.create_reject_data(x, proportion=1, method='uniform_hsphere',
                                          pca=True, pca_variance=0.99, pca_components=0,
                                          hshape_cov=0, hshape_prop_in=0.99,
                                          hshape_multiplier=1.5)
            n_samples = len(y)
            x_train = x[:int(n_samples/2),:]
            y_train = y[:int(n_samples/2)]
            r_train = r[:int(n_samples/2),:]

            fig = plt.figure('training_data')
            fig.clf()
            scatterplots.plot_data_and_reject(x_train,y_train,r_train,fig=fig)
            diary.save_figure(fig,'{}_training_data_synthetic_example.pdf'.format(example))

            # Classifier of reject data
            model_rej = train_reject_model(x_train, r_train)

            # Classifier of training data
            model_clas = train_classifier_model(x_train, y_train)

            # TRAINING SCORES
            c1_rs = model_rej.predict_proba(r_train)
            c1_ts = model_rej.predict_proba(x_train)
            c2_ts = model_clas.predict_proba(x_train)

            ace1, ace2, cace = compute_cace(c1_rs, c1_ts, c2_ts, y_train)

            print('TRAIN RESULTS')
            print('Step1 Average Cross-entropy = {}'.format(ace1))
            print('Step2 Average Cross-entropy = {}'.format(ace2))
            print('Composite Average Cross-entropy = {}'.format(cace))
            diary.add_entry('training', ['Example', example,
                                         'Iteration', iteration,
                                         'Step1 ACE', ace1])
            diary.add_entry('training', ['Example', example,
                                         'Iteration', iteration,
                                         'Step2 ACE', ace2])
            diary.add_entry('training', ['Example', example,
                                         'Iteration', iteration,
                                         'Composite ACE', cace])

            step1_reject_scores = c1_rs[:,1]
            step1_training_scores = c1_ts[:,1]
            step2_training_scores = c2_ts[:,1]
            training_labels = y_train

            print("Step1 Accuracy = {} (prior = {})".format(
                (np.sum(step1_reject_scores < 0.5) +
                 np.sum(step1_training_scores >=
                     0.5))/(np.alen(x_train)+np.alen(r_train)),
                 np.max([np.alen(x_train),np.alen(r_train)])/(np.alen(x_train)+np.alen(r_train))))

            print("Step2 Accuracy = {} (prior = {})".format(
                np.mean(np.argmax(c2_ts, axis=1) == y_train),
                np.max([1-np.mean(y_train), np.mean(y_train)])))


            #####################################################
            # VALIDATION                                        #
            #####################################################
            x_valid = x[int(n_samples/2):,:]
            y_valid = y[int(n_samples/2):]
            r_valid = r[int(n_samples/2):,:]

            fig = plt.figure('validation_data')
            fig.clf()
            scatterplots.plot_data_and_reject(x_valid,y_valid,r_valid,fig=fig)
            diary.save_figure(fig,'{}_validation_data_synthetic_example.pdf'.format(example))

            # TEST SCORES
            c1_rs = model_rej.predict_proba(r_valid)
            c1_ts = model_rej.predict_proba(x_valid)
            c2_ts = model_clas.predict_proba(x_valid)

            ace1, ace2, cace = compute_cace(c1_rs, c1_ts, c2_ts, y_valid)

            print('TEST RESULTS')
            print('Step1 Average Cross-entropy = {}'.format(ace1))
            print('Step2 Average Cross-entropy = {}'.format(ace2))
            print('Composite Average Cross-entropy = {}'.format(cace))
            diary.add_entry('validation', ['Example', example,
                                           'Iteration', iteration,
                                           'Step1 ACE', ace1])
            diary.add_entry('validation', ['Example', example,
                                           'Iteration', iteration,
                                           'Step2 ACE', ace2])
            diary.add_entry('validation', ['Example', example,
                                           'Iteration', iteration,
                                           'Composite ACE', cace])

            step1_reject_scores = c1_rs[:,1]
            step1_training_scores = c1_ts[:,1]
            step2_training_scores = c2_ts[:,1]
            training_labels = y_valid

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
                 np.sum(step1_training_scores >=
                     0.5))/(np.alen(x_valid)+np.alen(r_valid)),
                 np.max([np.alen(x_valid),np.alen(r_valid)])/(np.alen(x_valid)+np.alen(r_valid))))

            print("Step2 Accuracy = {} (prior = {})".format(
                np.mean(np.argmax(c2_ts, axis=1) == y_valid),
                np.max([1-np.mean(y_valid), np.mean(y_valid)])))

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
        diary.save_figure(fig,'{}_exp_rec_acc_synthetic_example.pdf'.format(example))

        # Area under the RGP curve
        rgp = RGP(step1_reject_scores, step1_training_scores,
                  step2_training_scores, training_labels)

        print("Area = {}".format(rgp.calculate_area()))
        fig = plt.figure('RGP_acc')
        fig.clf()
        rgp.plot(fig, accuracy=True)
        diary.save_figure(fig,'{}_rgp_acc_synthetic_example.pdf'.format(example))

        print("Area = {}".format(rgp.calculate_area()))
        fig = plt.figure('RGP_pre')
        fig.clf()
        rgp.plot(fig, precision=True)
        diary.save_figure(fig,'{}_rgp_pre_synthetic_example.pdf'.format(example))

        print("Area = {}".format(rgp.calculate_area()))
        fig = plt.figure('RGP')
        fig.clf()
        rgp.plot(fig)
        diary.save_figure(fig,'{}_rgp_synthetic_example.pdf'.format(example))

        ag = AbstainGainCurve(step1_reject_scores, step1_training_scores,
                  step2_training_scores, training_labels)

        print("Area = {}".format(ag.calculate_area()))
        fig = plt.figure('AG')
        fig.clf()
        ag.plot(fig)
        diary.save_figure(fig,'{}_ag_synthetic_example.pdf'.format(example))

        print("Optimal threshold for the first classifier = {}".format(rgp.get_optimal_step1_threshold()))

        print("RG1_PG1_ACC2")
        fig = plt.figure('RG1_PG1_ACC2')
        fig.clf()
        rgrpg = RGRPG(step1_reject_scores, step1_training_scores, step2_training_scores, training_labels)
        rgrpg.plot_simple_3d(fig)
        diary.save_figure(fig,'{}_rg1_pg1_acc2_synthetic_example.pdf'.format(example))

        print("ROC_curve_classifier_1")
        fig = plt.figure('ROC_curve_classifier_1')
        fig.clf()
        p1 = np.vstack([np.zeros((len(step1_reject_scores),1)),
                        np.ones((len(step1_training_scores),1))])
        fpr, tpr, thresholds_1 = roc_curve(p1, np.append(step1_reject_scores,
                                                       step1_training_scores))
        area = auc(fpr, tpr)
        print('AUC Classifier 1 = {}'.format(area))
        # FIXME there is a problem with some decision trees
        #print('PRGA Classifier 1 = {}'.format(rgrpg.prga))
        plt.plot(fpr, tpr, 'k.-')
        plt.xlabel('$FPr_1$')
        plt.ylabel('$TPr_1$')
        diary.save_figure(fig,'{}_roc_clas1_synthetic_example.pdf'.format(example))

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
        diary.save_figure(fig,'{}_log_loss_synthetic_example.pdf'.format(example))

        # Reducing the problem to three probabilities encoding:
        # Reject, Train*positive, Train*negative
        c1_rs = model_rej.predict_proba(r_valid)
        c1_ts = model_rej.predict_proba(x_valid)
        c2_rs = model_clas.predict_proba(r_valid)
        c2_ts = model_clas.predict_proba(x_valid)

        step1_reject_scores = c1_rs[:,1]
        step1_training_scores = c1_ts[:,1]
        step2_reject_scores = (c2_rs.T*step1_reject_scores).T
        step2_training_scores = (c2_ts.T*step1_training_scores).T

        # p = [R, +, -]
        # q = [R, T*+, T*-]
        q1 = np.expand_dims(np.vstack([c1_rs, c1_ts])[:,0], axis=1)
        p1 = np.vstack([np.ones((len(c1_rs),1)), np.zeros((len(c1_ts),1))])

        q2 = np.vstack([step2_reject_scores, step2_training_scores])
        p2 = label_binarize(y_valid, np.unique(y))
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
        diary.save_figure(fig,'{}_combined_accuracy_synthetic_example.pdf'.format(example))

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
        diary.save_figure(fig,'{}_jaccard_synthetic_example.pdf'.format(example))

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
        diary.save_figure(fig,'{}_fbeta_synthetic_example.pdf'.format(example))

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
        diary.save_figure(fig,'{}_pre_rec_synthetic_example.pdf'.format(example))


        if x_valid.shape[1] == 2:
            # FIXME take into account maximum values for training instances
            x_min = np.min(r_valid,axis=0)
            x_max = np.max(r_valid,axis=0)
            delta = 60
            x1_lin = np.linspace(x_min[0], x_max[0], delta)
            x2_lin = np.linspace(x_min[1], x_max[1], delta)

            MX1, MX2 = np.meshgrid(x1_lin, x2_lin)
            x_grid = np.asarray([MX1.flatten(),MX2.flatten()]).T
            q1_grid =  model_rej.predict_proba(x_grid)
            q2_grid =  model_clas.predict_proba(x_grid)

            q_grid = np.hstack([np.expand_dims(q1_grid[:,0], axis=1), q2_grid])

            # HEATMAP OF PROBABILITIES
            fig = plt.figure('heat_map_nw', frameon=False)
            plt.clf()
            heatmaps.plot_probabilities(q_grid)
            plt.title('Probabilities')
            diary.save_figure(fig,'{}_heat_map_nw_synthetic_example.pdf'.format(example))

            #q_grid = np.hstack([np.expand_dims(q1_grid[:,0], axis=1), (q2_grid.T*q1_grid[:,1]).T])

            # HEATMAP OF WEIGHTED PROBABILITIES
            #fig = plt.figure('heat_map_w', frameon=False)
            #plt.clf()
            #heatmaps.plot_probabilities(q_grid)
            #plt.title('Weighted probabilities')
            #diary.save_figure(fig,'{}_heat_map_w_synthetic_example.pdf'.format(example))

            # SCATTERPLOT OF PREDICTIONS
            threshold = thresholds_joint[-1-np.argmax(np.mean(fbeta, axis=1)[::-1])]
            predictions_grid = q_grid >= threshold
            fig = plt.figure('fbeta_grid')
            plt.clf()
            scatterplots.plot_predictions(x_grid, predictions_grid)
            plt.title('$F_{}$ optimal threshold = {}'.format("{"+str(beta)+"}", threshold))
            diary.save_figure(fig,'{}_fbeta_prediction_grid_synthetic_example.pdf'.format(example))

            # SCATTERPLOT OF PREDICTIONS
            threshold = thresholds_joint[-1-np.argmax(np.mean(jaccard, axis=1)[::-1])]
            predictions_grid = q_grid >= threshold
            fig = plt.figure('jaccard_grid')
            plt.clf()
            scatterplots.plot_predictions(x_grid, predictions_grid)
            plt.title('Jaccard optimal threshold = {}'.format(threshold))
            diary.save_figure(fig,'{}_jaccard_prediction_grid_synthetic_example.pdf'.format(example))

            # SCATTERPLOT OF PREDICTIONS
            threshold = thresholds_joint[-1-np.argmax(np.mean(accuracies_joint, axis=1)[::-1])]
            predictions_grid = q_grid >= threshold
            fig = plt.figure('accuracies_grid')
            plt.clf()
            scatterplots.plot_predictions(x_grid, predictions_grid)
            plt.title('Accuracy optimal threshold = {}'.format(threshold))
            diary.save_figure(fig,'{}_accuracies_prediction_grid_synthetic_example.pdf'.format(example))

            # SCATTERPLOT OF ARGMAX PREDICTIONS
            predictions_grid = label_binarize(q_grid.argmax(axis=1),
                    range(q_grid.shape[1])).astype('bool')
            fig = plt.figure('argmax_grid')
            plt.clf()
            scatterplots.plot_predictions(x_grid, predictions_grid)
            plt.title('Argmax')
            diary.save_figure(fig,'{}_argmax_prediction_grid_synthetic_example.pdf'.format(example))

            # CONTOUR CLASSIFIER 1
            fig = plt.figure('validation_data')
            x_min = np.min(r_valid,axis=0)
            x_max = np.max(r_valid,axis=0)
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
            plt.title('Reject model contour lines')
            diary.save_figure(fig,'{}_synthetic_example_reject_contour.pdf'.format(example))
