from __future__ import division

import numpy as np
np.random.seed(42)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.ion()
plt.rcParams['figure.figsize'] = (5,4)
plt.rcParams['figure.autolayout'] = True

from sklearn.mixture import GMM
from sklearn import svm
from sklearn import tree

from cwc.synthetic_data import toy_examples
from cwc.synthetic_data import reject
from cwc.evaluation.rgrpg import RGRPG
from cwc.evaluation.rgp import RGP
from cwc.evaluation.abstaingaincurve import AbstainGainCurve

def generate_data(d3=False):
    if d3:
        samples = [500,         # Class 1
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
    else:
        samples = [500,         # Class 1
                   200]         # Class 2
        means = [[2,2],       # Class 1
                 [5,5]]       # Class 2
        covs = [[[2,-1],       # Class 1
                 [-1,1]],
                [
                 [1,1],       # Class 2
                 [1,2]]]

    x, y = toy_examples.generate_gaussians(means=means, covs=covs,
                                           samples=samples)
    return x, y


def plot_data_and_reject(x, y, r):
    # Options for plotting
    colors = ['r', 'b', 'y']    # One color per class + reject

    fig = plt.figure('data_reject')
    fig.clf()
    if x.shape[1] >= 3:
        ax = fig.add_subplot(111, projection='3d')
        for k, c in enumerate(colors[:-1]):
            index = (y == k)
            ax.scatter(x[index,0], x[index,1], x[index,2], c=c)
        ax.scatter(r[:,0], r[:,1], r[:,2], marker='x', c=colors[-1])
    else:
        ax = fig.add_subplot(111)
        for k, c in enumerate(colors[:-1]):
            index = (y == k)
            ax.scatter(x[index,0], x[index,1], c=c)
        ax.scatter(r[:,0], r[:,1], marker='x', c=colors[-1])


def train_reject_model(x, r):
    """Train a classifier of training points

    Returns a classifier that predicts high probability values for training
    points and low probability values for reject points.
    """
    # model_rej = svm.SVC(probability=True)
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
    x, y = generate_data(d3=False)
    r = reject.create_reject_data(x, proportion=1, method='uniform_hsphere',
                                  pca=True, pca_variance=0.99, pca_components=0,
                                  hshape_cov=0, hshape_prop_in=0.97,
                                  hshape_multiplier=1.2)
    plot_data_and_reject(x,y,r)

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

    ag = AbstainGainCurve(step1_reject_scores, step1_training_scores,
              step2_training_scores, training_labels)

    print("Area = {}".format(ag.calculate_area()))
    fig = plt.figure('AG')
    fig.clf()
    ag.plot(fig)

    print("Optimal threshold for the first classifier = {}".format(rgp.get_optimal_step1_threshold()))
