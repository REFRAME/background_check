import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.ion()

from sklearn.mixture import GMM
from sklearn import tree

from cwc.synthetic_data import toy_examples
from cwc.synthetic_data import reject
from cwc.evaluation.rgrpg import RGRPG

def generate_data():
    samples = [700,         # Class 1
               400]         # Class 2
    means = [[3,3,3],       # Class 1
             [3,5,5]]       # Class 2
    covs = [[[2,1,0],       # Class 1
             [1,2,0],
             [0,0,1]],
            [
             [1,0,0],       # Class 2
             [0,1,1],
             [0,1,2]]]

    x, y = toy_examples.generate_gaussians(means=means, covs=covs,
                                           samples=samples)
    return x, y


def generate_reject_data(x):
    # Options for reject data generation
    method = 'uniform_hsphere'
    proportion = 0.9
    hshape_cov = 0
    hshape_prop_in = 0.99
    hshape_multiplier = 1.5
    pca = True
    pca_components = 0
    pca_variance = 0.9
    r = reject.create_reject_data(x, proportion=proportion,
                                  method=method, pca=pca,
                                  pca_variance=pca_variance,
                                  pca_components=pca_components,
                                  hshape_cov=hshape_cov,
                                  hshape_prop_in=hshape_prop_in,
                                  hshape_multiplier=hshape_multiplier)
    return r

def plot_data_and_reject(x,y,r):
    # Options for plotting
    colors = ['r', 'b', 'y']    # One color per class + reject

    fig = plt.figure('data_reject')
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')
    for k, c in enumerate(colors[:-1]):
        index = (y == k)
        ax.scatter(x[index,0], x[index,1], x[index,2], c=c)
    ax.scatter(r[:,0], r[:,1], r[:,2], c=colors[-1])


def train_reject_model(x,r):
    # TODO compare with a reject model that is not generative
    n_components=10
    covariance_type='diag'
    random_state=42
    thresh=None
    tol=0.001
    min_covar=0.001
    n_iter=100
    n_init=1
    params='wmc'
    init_params='wmc'
    verbose=0
    model_rej = GMM(n_components=n_components, covariance_type=covariance_type,
                    random_state=random_state, thresh=thresh, tol=tol,
                    min_covar=min_covar, n_iter=n_iter, n_init=n_init,
                    params=params, init_params=init_params, verbose=verbose)
    model_rej.fit(x)
    return model_rej

def train_classifier_model(x,y):
    model_clas = tree.DecisionTreeClassifier(max_depth=2)
    model_clas = model_clas.fit(x,y)
    return model_clas

if __name__ == "__main__":
    x,y = generate_data()
    r = generate_reject_data(x)
    plot_data_and_reject(x,y,r)

    # Classifier of reject data
    model_rej = train_reject_model(x,r)

    # Classifier of training data
    model_clas = train_classifier_model(x,y)

    # Get scores
    step1_reject_scores = model_rej.predict_proba(r).max(axis=1)
    step1_training_scores = model_rej.predict_proba(x).max(axis=1)
    step2_training_scores = model_clas.predict_proba(x)[:,0]
    training_labels = y

    # Evaluate models
    rgrpg = RGRPG(step1_reject_scores, step1_training_scores,
                  step2_training_scores, training_labels)

    print rgrpg.calculate_volume()
    plt.figure()
    rgrpg.plot_rgrpg_2d()
    rgrpg.plot_rgrpg_3d()
