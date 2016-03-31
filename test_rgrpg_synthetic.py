from __future__ import division

import numpy as np
np.random.seed(42)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.ion()
plt.rcParams['figure.figsize'] = (4,3)
plt.rcParams['figure.autolayout'] = True

from sklearn.mixture import GMM
from sklearn import svm
from sklearn import tree

from cwc.synthetic_data import toy_examples
from cwc.synthetic_data import reject
from cwc.evaluation.rgrpg import RGRPG
from cwc.evaluation.rgp import RGP
from cwc.evaluation.abstaingaincurve import AbstainGainCurve

# FIXME change the holes for optional arguments
def generate_data(example=1):
    if example == 1:
        holes=[2,1]
        samples = [600,         # Class 1
                   400]         # Class 2
        means = [[2,2,2],       # Class 1
                 [3,3,4]]       # Class 2
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
                 [0,-10],       # Class 2
                 [7,4]]       # Class 3
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
    else:
        raise Exception('Example {} does not exist'.format(example))

    x, y = toy_examples.generate_gaussians(means=means, covs=covs,
                                           samples=samples, holes=holes)
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


def train_reject_model(x, r):
    """Train a classifier of training points

    Returns a classifier that predicts high probability values for training
    points and low probability values for reject points.
    """
    #model_rej = svm.SVC(probability=True)
    model_rej = tree.DecisionTreeClassifier(max_depth=8)

    xr = np.vstack((x,r))
    y = np.hstack((np.ones(np.alen(x)), np.zeros(np.alen(r)))).T
    model_rej.fit(xr, y)

    return model_rej


def train_classifier_model(x,y):
    #model_clas = svm.SVC(probability=True)
    model_clas = tree.DecisionTreeClassifier(max_depth=5)
    model_clas = model_clas.fit(x,y)
    return model_clas

if __name__ == "__main__":
    for i in range(1,4):
        x, y = generate_data(i)
        r = reject.create_reject_data(x, proportion=1, method='uniform_hsphere',
                                      pca=True, pca_variance=0.99, pca_components=0,
                                      hshape_cov=0, hshape_prop_in=0.97,
                                      hshape_multiplier=1.2)

        fig = plt.figure('data_reject')
        fig.clf()
        plot_data_and_reject(x,y,r,fig)
        fig.savefig('{}_synthetic_example.pdf'.format(i))

        # Classifier of reject data
        model_rej = train_reject_model(x, r)

        # Classifier of training data
        model_clas = train_classifier_model(x, y)

        # Get scores
        step1_reject_scores = model_rej.predict_proba(r)[:,1]
        step1_training_scores = model_rej.predict_proba(x)[:,1]
        step2_training_scores = model_clas.predict_proba(x)[:,1]
        training_labels = y

        # Show scores
        fig = plt.figure('scores')
        fig.clf()
        ax = fig.add_subplot(111)
        ax.set_title('Scores')
        ax.hist([step1_reject_scores, step1_training_scores,
                 step2_training_scores])
        ax.legend(['step1_reject', 'step1_training', 'step2_class1'],
                  loc='upper center')

        # Evaluate models
        # Accuracy
        print("Step1 Accuracy = {} (prior = {})".format(
            (np.sum(step1_reject_scores < 0.5) +
             np.sum(step1_training_scores >= 0.5))/(np.alen(x)+np.alen(r)),
             np.max([np.alen(x),np.alen(r)])/(np.alen(x)+np.alen(r))))

        print("Step2 Accuracy = {} (prior = {})".format(
            np.mean((step2_training_scores >= 0.5) == y),
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

        # Area under the RGP curve
        rgp = RGP(step1_reject_scores, step1_training_scores,
                  step2_training_scores, training_labels)

        print("Area = {}".format(rgp.calculate_area()))
        fig = plt.figure('RGP')
        fig.clf()
        rgp.plot(fig)
        fig.savefig('{}_rgp_synthetic_example.pdf'.format(i))

        ag = AbstainGainCurve(step1_reject_scores, step1_training_scores,
                  step2_training_scores, training_labels)

        print("Area = {}".format(ag.calculate_area()))
        fig = plt.figure('AG')
        fig.clf()
        ag.plot(fig)
        fig.savefig('{}_ag_synthetic_example.pdf'.format(i))

        print("Optimal threshold for the first classifier = {}".format(rgp.get_optimal_step1_threshold()))
