from __future__ import division

import numpy as np
np.random.seed(42)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# plt.ion()
plt.rcParams['figure.figsize'] = (7, 6)
plt.rcParams['figure.autolayout'] = True

from sklearn.mixture import GMM
from sklearn import svm
from sklearn import tree
from sklearn.preprocessing import label_binarize

from cwc.synthetic_data import toy_examples
from cwc.synthetic_data import reject
from cwc.visualization import heatmaps
from cwc.visualization import scatterplots

# FIXME change the holes for optional arguments
def generate_data(example=1, hole_centers=None):
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
    elif example == 8:
        holes=[3,2,1,2]
        samples = [400,         # Class 1
                   300,         # Class 2
                   300,         # Class 3
                   150]         # Class 4
        means = [[-3,0],       # Class 1
                 [2,2],       # Class 2
                 [-3,0],       # Class 3
                 [3,-4]]       # Class 4
        covs = [[[3,0.3],       # Class 1
                 [0.3,2]],
                [[3,0],       # Class 2
                 [0,3]],
                [[1,-0.2],       # Class 3
                 [-0.2,1]],
                [[2,-0.5],       # Class 4
                 [-0.5,2]]]
    else:
        raise Exception('Example {} does not exist'.format(example))

    x, y, hole_centers = toy_examples.generate_gaussians(
                          means=means, covs=covs, samples=samples, holes=holes,
                            hole_centers=hole_centers)
    if example==5:
        y[y==2] = 1
    return x, y, hole_centers


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
    for example in [3]:
        np.random.seed(42)
        print('Running example = {}'.format(example))

        #####################################################
        # TRAINING                                          #
        #####################################################
        x, y, hole_centers = generate_data(example)
        r = reject.create_reject_data(x, proportion=1, method='uniform_hsphere',
                                      pca=True, pca_variance=0.99, pca_components=0,
                                      hshape_cov=0, hshape_prop_in=0.99,
                                      hshape_multiplier=1.5)

        fig = plt.figure('training_data')
        fig.clf()
        scatterplots.plot_data_and_reject(x,y,r,fig)
        fig.savefig('{}_training_data_synthetic_example.pdf'.format(example))

        # Classifier of training data
        model_clas = train_classifier_model(x, y)

        model_clas_ts = model_clas.predict_proba(x)
        model_clas_rs = model_clas.predict_proba(r)

        p_x = np.hstack([model_clas_ts, x])
        p_r = np.hstack([model_clas_rs, r])

        # Classifier of reject data
        model_rej = train_reject_model(p_x, p_r)

        #####################################################
        # VALIDATION                                        #
        #####################################################
        # FIXME the validation data does not have the hole in the same
        # position as the training data
        x, y, hole_centers = generate_data(example, hole_centers)
        r = reject.create_reject_data(x, proportion=1, method='uniform_hsphere',
                                      pca=True, pca_variance=0.99, pca_components=0,
                                      hshape_cov=0, hshape_prop_in=0.99,
                                      hshape_multiplier=1.5)

        fig = plt.figure('validation_data')
        fig.clf()
        scatterplots.plot_data_and_reject(x,y,r,fig)
        fig.savefig('{}_validation_data_synthetic_example.pdf'.format(example))

        # TEST SCORES
        model_clas_ts = model_clas.predict_proba(x)
        model_clas_rs = model_clas.predict_proba(r)

        p_x = np.hstack([model_clas_ts, x])
        p_r = np.hstack([model_clas_rs, r])

        model_rej_rs = model_rej.predict_proba(p_r)
        model_rej_ts = model_rej.predict_proba(p_x)

        # Reducing the problem to three probabilities encoding:
        # Reject, Train*positive, Train*negative
        # c1_rs = model_rej.predict_proba(r)
        # c1_ts = model_rej.predict_proba(x)
        # c2_rs = model_clas.predict_proba(r)
        # c2_ts = model_clas.predict_proba(x)
        #
        # step1_reject_scores = c1_rs[:,1]
        # step1_training_scores = c1_ts[:,1]
        # step2_reject_scores = (c2_rs.T*step1_reject_scores).T
        # step2_training_scores = (c2_ts.T*step1_training_scores).T

        # p = [R, +, -]
        # q = [R, T*+, T*-]
        # q1 = np.expand_dims(np.vstack([c1_rs, c1_ts])[:, 0], axis=1)
        # p1 = np.vstack([np.ones((len(c1_rs), 1)), np.zeros((len(c1_ts), 1))])
        #
        # q2 = np.vstack([step2_reject_scores, step2_training_scores])
        # p2 = label_binarize(y, np.unique(y))
        # label binarize creates [n_samples, n_classes] for n_classes > 2
        # and [n_samples, 1] for n_classes = 2
        # if p2.shape[1] == 1:
        #     p2 = np.hstack([1-p2, p2])
        # p2 = np.vstack([np.zeros((len(c1_rs), p2.shape[1])), p2])
        #
        # q = np.hstack([q1, q2])
        # p = np.hstack([p1, p2])
        # print average_cross_entropy(p,q)
        #
        # thresholds_joint = np.linspace(0,1,100)
        # accuracies_joint = np.empty((len(thresholds_joint), p.shape[1]))
        # for i, threshold in enumerate(thresholds_joint):
        #     prediction = (q >= threshold)
        #     #prediction[:,0] = 1-np.sum(prediction[:,1:], axis=1)
        #     accuracies_joint[i] = np.mean(prediction == p, axis=0)
        #
        # fig = plt.figure('combined_accuracy')
        # plt.clf()
        # plt.plot(thresholds_joint, np.mean(accuracies_joint, axis=1), '.-', label='Model_1+2')
        # plt.plot(thresholds_joint, accuracies_joint, '.-')
        # n_instances = np.sum(p, axis=0)
        # n_classes = p.shape[1]
        # for c_id in range(n_classes):
        #     n_at = n_instances[c_id]
        #     n_nat = sum(n_instances)-n_at
        #     c_prop = (n_classes*n_at + n_nat)/(n_classes*sum(n_instances))
        #     plt.plot([0,1], [c_prop, c_prop], label='prop.{}'.format(c_id))
        # plt.xlabel('threshold')
        # plt.ylabel('Accuracy')
        # plt.legend(loc='lower right')
        # plt.ylim([0,1])
        # fig.savefig('{}_combined_accuracy_synthetic_example.pdf'.format(example))
        #
        # thresholds_joint = np.linspace(0,1,100)
        # tp_joint = np.empty((len(thresholds_joint), p.shape[1]))
        # fp_joint = np.empty((len(thresholds_joint), p.shape[1]))
        # fn_joint = np.empty((len(thresholds_joint), p.shape[1]))
        # for i, threshold in enumerate(thresholds_joint):
        #     prediction = (q >= threshold)
        #     tp_joint[i] = np.mean(np.logical_and(prediction, p), axis=0)
        #     fp_joint[i] = np.mean(np.logical_and(prediction, 1-p), axis=0)
        #     fn_joint[i] = np.mean(np.logical_and(1-prediction, p), axis=0)
        #
        # jaccard = tp_joint/(tp_joint + fp_joint + fn_joint)
        # fig = plt.figure('jaccard')
        # plt.clf()
        # plt.plot(thresholds_joint, np.mean(jaccard, axis=1), '.-', label='Multi-jaccard')
        # plt.plot(thresholds_joint, jaccard[:,0], '.-', label='Reject')
        # for id_c in range(1,jaccard.shape[1]):
        #     plt.plot(thresholds_joint, jaccard[:,id_c], '.-', label='Class {}'.format(id_c))
        # plt.xlabel('threshold')
        # plt.ylabel('Jaccard')
        # plt.legend(loc='bottom-left')
        # fig.savefig('{}_jaccard_synthetic_example.pdf'.format(example))
        #
        # beta = 1
        # fbeta = ((1+beta**2)*tp_joint)/((1+beta**2)*tp_joint + beta**2*fn_joint + fp_joint)
        # fig = plt.figure('fbeta')
        # plt.clf()
        # plt.plot(thresholds_joint, np.mean(fbeta, axis=1), '.-', label='Multi-fbeta')
        # plt.plot(thresholds_joint, fbeta[:,0], '.-', label='Reject')
        # for id_c in range(1,fbeta.shape[1]):
        #     plt.plot(thresholds_joint, fbeta[:,id_c], '.-', label='Class {}'.format(id_c))
        # plt.xlabel('threshold')
        # plt.ylabel('fbeta')
        # plt.legend(loc='bottom-left')
        # fig.savefig('{}_fbeta_synthetic_example.pdf'.format(example))
        #
        # precision = tp_joint/(tp_joint + fp_joint)
        # recall = tp_joint/(tp_joint + fn_joint)
        # fig = plt.figure('precision_recall')
        # plt.clf()
        # plt.plot(np.mean(recall, axis=1), np.mean(precision, axis=1), '.-',
        #         label='Multi-pre-rec')
        # plt.plot(recall[:,0], precision[:,0], '.-', label='Reject')
        # for id_c in range(1,recall.shape[1]):
        #     plt.plot(recall[:,id_c], precision[:,id_c], '.-', label='Class {}'.format(id_c))
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.ylim([0,1])
        # plt.xlim([0,1])
        # plt.legend(loc='bottom-left')
        # fig.savefig('{}_pre_rec_synthetic_example.pdf'.format(example))


        if x.shape[1] == 2:
            # FIXME take into account maximum values for training instances
            x_min = np.min(r, axis=0)
            x_max = np.max(r, axis=0)
            delta = 70
            x1_lin = np.linspace(x_min[0], x_max[0], delta)
            x2_lin = np.linspace(x_min[1], x_max[1], delta)

            MX1, MX2 = np.meshgrid(x1_lin, x2_lin)
            x_grid = np.asarray([MX1.flatten(), MX2.flatten()]).T

            model_clas_grid = model_clas.predict_proba(x_grid)

            p_x_grid = np.hstack([model_clas_grid, x_grid])

            model_rej_grid = model_rej.predict_proba(p_x_grid)

            q_grid = label_binarize(np.argmax(model_clas_grid, axis=1),
                                    np.arange(model_clas_grid.shape[1]))
            if model_clas_grid.shape[1] == 2:
                q_grid = np.hstack([1-q_grid, q_grid])

            q_grid = np.hstack([1-np.argmax(model_rej_grid, axis=1).reshape(
                -1, 1), q_grid]).astype(bool)
            # q_grid = np.hstack([np.expand_dims(q1_grid[:,0], axis=1), (q2_grid.T*q1_grid[:,1]).T])

            # # HEATMAP OF PROBABILITIES
            # fig = plt.figure('heat_map', frameon=False)
            # plt.clf()
            # heatmaps.plot_probabilities(q_grid)
            # plt.title('Weighted probabilities')
            # fig.savefig('{}_heat_map_synthetic_example.pdf'.format(example))

            # SCATTERPLOT OF PREDICTIONS
            # threshold = thresholds_joint[-1-np.argmax(np.mean(fbeta, axis=1)[::-1])]
            predictions_grid = q_grid
            fig = plt.figure('multi-target grid')
            plt.clf()
            scatterplots.plot_predictions(x_grid, predictions_grid)
            plt.title('Multi-target grid')
            plt.show()
            # fig.savefig('{}_fbeta_prediction_grid_synthetic_example.pdf'.format(example))

            # SCATTERPLOT OF PREDICTIONS
            # threshold = thresholds_joint[-1-np.argmax(np.mean(jaccard, axis=1)[::-1])]
            # predictions_grid = q_grid >= threshold
            # fig = plt.figure('jaccard_grid')
            # plt.clf()
            # scatterplots.plot_predictions(x_grid, predictions_grid)
            # plt.title('Jaccard optimal threshold = {}'.format(threshold))
            # fig.savefig('{}_jaccard_prediction_grid_synthetic_example.pdf'.format(example))

            # SCATTERPLOT OF PREDICTIONS
            # threshold = thresholds_joint[-1-np.argmax(np.mean(accuracies_joint, axis=1)[::-1])]
            # predictions_grid = q_grid >= threshold
            # fig = plt.figure('accuracies_grid')
            # plt.clf()
            # scatterplots.plot_predictions(x_grid, predictions_grid)
            # plt.title('Accuracy optimal threshold = {}'.format(threshold))
            # fig.savefig('{}_accuracies_prediction_grid_synthetic_example.pdf'.format(example))


            # CONTOUR CLASSIFIER 1
            # fig = plt.figure('validation_data')
            # x_min = np.min(r,axis=0)
            # x_max = np.max(r,axis=0)
            # x1_lin = np.linspace(x_min[0], x_max[0], delta)
            # x2_lin = np.linspace(x_min[1], x_max[1], delta)
            #
            # MX1, MX2 = np.meshgrid(x1_lin, x2_lin)
            # x_grid = np.asarray([MX1.flatten(),MX2.flatten()]).T
            # p_grid =  model_rej.predict_proba(x_grid)
            # levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            # CS = plt.contour(x_grid[:,0].reshape(delta,delta),
            #                  x_grid[:,1].reshape(delta,delta),
            #                  p_grid[:,1].reshape(delta,-1), levels, linewidths=3,
            #                  alpha=0.5)
            # plt.clabel(CS, fontsize=15, inline=2)
            # plt.title('Reject model contour lines')
            # fig.savefig('{}_synthetic_example_reject_contour.pdf'.format(example))
