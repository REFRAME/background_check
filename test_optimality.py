from __future__ import division
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from cwc.models.density_estimators import MyMultivariateNormal

import pandas as pd

np.random.seed(42)
plt.ion()
plt.rcParams['figure.figsize'] = (7,4)
plt.rcParams['figure.autolayout'] = True

colors = ['red', 'blue', 'white']

def normalized_brier_score(y, prediction):
    return ((prediction/np.max(prediction) - y)**2).mean()

def one_vs_rest_roc_curve(y,p, pos_label=0):
    """Returns the roc curve of class 0 vs the rest of the classes"""
    aux = np.zeros_like(y)
    aux[y!=pos_label] = 1
    return roc_curve(aux, p, pos_label=pos_label)


def convex_hull(points):
    """Computes the convex hull of a set of 2D points.
    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    Source code from:
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return upper


def plot_data(x,y, fig=None, title=None):
    if fig is None:
        fig = plt.figure('Data')
    fig.clf()
    ax = fig.add_subplot(111)
    classes = np.unique(y)
    for c in classes:
        ax.scatter(x[y==c,0], x[y==c,1], c=colors[c], label='Class {}'.format(c))
    ax.legend()


def plot_data_and_contourlines(x,y,x_grid,ps, delta=50, fig=None, title=None):
    if fig is None:
        fig = plt.figure('gaussians')
    fig.clf()
    plot_data(x,y,fig=fig)

    ax = fig.add_subplot(111)
    # HEATMAP OF PROBABILITIES
    #levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # TODO Change colors of the contour lines to the matching class
    if len(ps) == 1:
        cmaps= ['jet']
    elif len(ps) == 2:
        cmaps= ['autumn_r', 'winter_r']
    else:
        cmaps= ['jet']*len(ps)
    for i, p in enumerate(ps):
        CS = ax.contour(x_grid[:,0].reshape(delta,delta),
                         x_grid[:,1].reshape(delta,delta),
                         p.reshape(delta,-1), linewidths=3,
                         alpha=1.0, cmap=cmaps[i]) # jet
        ax.clabel(CS, fontsize=20, inline=2)

    if title is not None:
        ax.set_title(title)


def get_grid(x, delta=50):
    x_min = np.min(x,axis=0)
    x_max = np.max(x,axis=0)

    x1_lin = np.linspace(x_min[0], x_max[0], delta)
    x2_lin = np.linspace(x_min[1], x_max[1], delta)

    MX1, MX2 = np.meshgrid(x1_lin, x2_lin)
    x_grid = np.asarray([MX1.flatten(),MX2.flatten()]).T

    return x_grid


def plot_roc_curve(y,p,fig=None,title='', pos_label=0):
    if fig is None:
        fig = plt.figure('roc_curve')
    fig.clf()

    roc = one_vs_rest_roc_curve(y, p, pos_label=pos_label)
    auroc = auc(roc[0], roc[1])
    ax = fig.add_subplot(111)
    ax.plot(roc[0], roc[1])
    ax.plot([0,1],[0,1], 'g--')
    upper_hull = convex_hull(zip(roc[0],roc[1]))
    rg_hull, pg_hull = zip(*upper_hull)
    plt.plot(rg_hull, pg_hull, 'r--')
    ax.set_title('{0} {1:.3f}'.format(title, auroc))
    ax.set_ylim([0, 1.01])
    ax.set_xlim([-0.01, 1.01])
    ax.grid(True)

    return auroc


def plot_cost_curve(y,p,fig=None,title='', pos_label=0):
    if fig is None:
        fig = plt.figure('roc_curve')
    fig.clf()


class BackgroundClass(object):
    def __init__(self, i=1):
        self.n_samples, self.mean, self.cov = self.background_description(i)
        self.mvn = MyMultivariateNormal(self.mean, self.cov)

    def set_experiment(self, i):
        self.__init__(i)

    def sample(self, n_samples=None):
        if n_samples is not None:
            return self.mvn.sample(n_samples)
        return self.mvn.sample(self.n_samples)

    @property
    def n_experiments(self):
        return len(self.description)

    @property
    def experiment_ids(self):
        return range(self.n_experiments)

    def background_description(self, i, samples=500):
        self.description = [(0,       [0,0],   [[1,0], [0,1]]),
                            (samples, [-1,-1], [[1,0], [0,1]]),
                            (samples, [0,0],   [[1,0], [0,1]]),
                            (samples, [1,1],   [[1,0], [0,1]]),
                            (samples, [-4,2],  [[1,0], [0,1]]),
                            (samples, [-3,3],  [[1,0], [0,1]]),
                            (samples, [-2,4],  [[1,0], [0,1]]),
                            (samples, [0,6],   [[1,0], [0,1]]),
                            (samples, [-6,0],  [[1,0], [0,1]])]
        if i >= len(self.description):
            raise Exception('Unknown background description id')
        return self.description[i]


class MyDataFrame(pd.DataFrame):
    def append_rows(self, rows):
        dfaux = pd.DataFrame(rows, columns=self.columns)
        return self.append(dfaux, ignore_index=True)


def main(pos_labels=[0,1], experiment_ids='all'):
    np.random.seed(42)

    # Columns for the DataFrame
    columns=['Experiment', 'Pos_label', 'Method', 'AUC']
    # Create a DataFrame to record all intermediate results
    df = MyDataFrame(columns=columns)

    # Two original classes
    samples = np.array([500,         # Class 0
                        500])         # Class 1
    means = np.array([[-1,-1],       # Class 1
                      [1,1]])       # Class 2
    covs = np.array([[[1,0],       # Class 1
                      [0,1]],
                     [[1,0],       # Class 2
                      [0,1]]])

    mvn = {}
    x = {}
    x_samples = []
    y_samples = []
    for i in range(len(means)):
        mvn[i] = MyMultivariateNormal(means[i], covs[i])
        x_samples.append(mvn[i].sample(samples[i]))
        y_samples.append(np.ones(samples[i])*i)

    bg_class = BackgroundClass()
    if experiment_ids == 'all':
        experiment_ids = bg_class.experiment_ids
    for bg_class_id in experiment_ids:
        x = x_samples[:]
        y = y_samples[:]

        # Background
        bg_class.set_experiment(bg_class_id)
        x.append(bg_class.sample())
        y.append(np.ones(bg_class.n_samples)*(len(means)))

        x = np.vstack(x)
        y = np.hstack(y).astype(int)

        x_grid = get_grid(x)

        p_grid = []
        for key in mvn.keys():
            p_grid.append(mvn[key].score(x_grid))

        p_grid = np.array(p_grid).T
        # Density estimation
        fig = plt.figure('Density')
        plot_data_and_contourlines(x,y,x_grid,[p_grid[:,0],p_grid[:,1]],fig=fig, title='Density')

        # Bayes
        prior = samples/sum(samples)
        P_x_t = np.sum(p_grid*prior, axis=1)
        posterior = (p_grid*prior)/P_x_t[:,None]

        fig = plt.figure('Bayes')
        plot_data_and_contourlines(x,y,x_grid,[posterior[:,0], posterior[:,1]], fig=fig, title='Bayes optimal')

        # Background check
        c2 = 1.0
        c1 = 0.01
        P_t = 1.0 - c1 # np.in1d(y,[0,1]).sum()/len(y)
        P_b = c1
        # FIXME look if the priors where right
        max_value = np.maximum(mvn[0].score([means[0]])*prior[0] + mvn[1].score([means[0]])*prior[0],
                               mvn[0].score([means[1]])*prior[1] + mvn[1].score([means[1]])*prior[1])
        P_x_b = max_value-c2*P_x_t

        # Probability of training and class given x
        numerator = p_grid*prior*P_t
        denominator = np.sum(np.hstack([numerator,
            (P_x_b*P_b)[:,None]]), axis=1)
        P_t_x = numerator/denominator[:,None]

        fig = plt.figure('Background check')
        plot_data_and_contourlines(x,y,x_grid,[P_t_x[:,0],P_t_x[:,1]], fig=fig,
                                   title='Background check')

        # SVC RBF
        x_train = np.vstack(x_samples)
        y_train = np.hstack(y_samples).astype(int)

        svc = svm.SVC(probability=True, kernel='linear')
        svc.fit(x_train,y_train)
        svc_pred = svc.predict_proba(x_grid)
        fig = plt.figure('svm')
        plot_data_and_contourlines(x,y,x_grid,[svc_pred[:,0],svc_pred[:,1]],
                                   fig=fig, title='P_SVC')

        svc_rbf = svm.SVC(probability=True, kernel='rbf', gamma='auto')
        svc_rbf.fit(x_train,y_train)
        svc_rbf_pred = svc_rbf.predict_proba(x_grid)
        fig = plt.figure('svm_rbf')
        plot_data_and_contourlines(x,y,x_grid,[svc_rbf_pred[:,0],
                                   svc_rbf_pred[:,1]], fig=fig, title='P_SVC_rbf')

        svc_poly = svm.SVC(probability=True, kernel='poly', gamma='auto',
                           degree=3)
        svc_poly.fit(x_train,y_train)
        svc_poly_pred = svc_poly.predict_proba(x_grid)
        fig = plt.figure('svm_poly')
        plot_data_and_contourlines(x,y,x_grid,[svc_poly_pred[:,0],
                                   svc_poly_pred[:,1]], fig=fig, title='P_SVC_poly')

        # Get the predictions for all the models
        predictions = {}
        P_x_t_y = []
        for key in mvn.keys():
            P_x_t_y.append(mvn[key].score(x))
        P_x_t_y = np.vstack(P_x_t_y).T
        predictions['Density'] = P_x_t_y
        P_y_x = P_x_t_y/np.sum(P_x_t_y, axis=1)[:,None]
        predictions['Bayes_optimal'] = P_y_x

        svc_p_y_x = svc.predict_proba(x)
        predictions['SVC_linear'] = svc_p_y_x
        svc_rbf_p_y_x = svc_rbf.predict_proba(x)
        predictions['SVC_rbf'] = svc_rbf_p_y_x
        svc_poly_p_y_x = svc_poly.predict_proba(x)
        predictions['SVC_poly'] = svc_poly_p_y_x

        # Background check
        P_x_t = np.sum(P_x_t_y*prior, axis=1)
        P_x_b = max_value-c2*P_x_t
        numerator = P_x_t_y*prior*P_t
        denominator = np.sum(numerator, axis=1) + P_x_b*P_b
        P_t_y_x = numerator/denominator[:,None]
        predictions['Background_check'] = P_t_y_x

        for pos_label in pos_labels:
            for method, prediction in predictions.iteritems():
                fig = plt.figure('roc_{}'.format(method))
                auroc = plot_roc_curve(y, prediction[:,pos_label], fig=fig,
                                       title='ROC {}'.format(method),
                                       pos_label=pos_label)
                df = df.append_rows([[bg_class_id, pos_label, method, auroc]])

                roc = one_vs_rest_roc_curve(y, prediction[:,pos_label],
                                            pos_label=pos_label)
                ## Q = np.zeros(len(roc[0]))
                ## cs = [1,1]
                ## c = cs[0]/sum(cs)
                ## # FIXME look at the priors, they do not include the background
                ## for i, (fpr, tpr, threshold) in enumerate(zip(roc[0], roc[1], roc[2])):
                ##     Q[i] = 2*(c*prior[0]*(1-tpr) + (1-c)*prior[1]*fpr)
                ## fig = plt.figure('cost_{}'.format(method))
                ## fig.clf()
                ## ax = fig.add_subplot(111)
                ## ax.plot(roc[2][1:], Q[1:])
                ## ax.set_xlabel('threshold')
                ## ax.set_ylabel('$Q_{cost}$')

                ##fig = plt.figure('cost_lines_{}'.format(method))
                ##fig.clf()
                ##ax = fig.add_subplot(111)
                ### FIXME look at the priors, they do not include the background
                ##Q_min = np.zeros(len(roc[0]))
                ##Q_max = np.zeros(len(roc[0]))
                ##for i, (fpr, tpr, threshold) in enumerate(zip(roc[0], roc[1], roc[2])):
                ##    Q_min[i] = 2*(prior[1]*fpr)
                ##    Q_max[i] = 2*(prior[0]*(1-tpr))
                ##    ax.plot([0, 1], [Q_min[i], Q_max[i]], '--', c='0.75')
                ##ax.set_xlabel('cost proportion')
                ##ax.set_ylabel('$Q_{cost}$')

                # FIXME look why the values for Q_skew are never larger than 1
                fig = plt.figure('skew_lines_{}'.format(method))
                fig.clf()
                ax = fig.add_subplot(111)
                Q_min = roc[0]
                Q_max = 1-roc[1]
                ax.plot(np.vstack((np.zeros_like(Q_min), np.ones_like(Q_max))), 
                        np.vstack((Q_min, Q_max)), '--', c='0.80')
                ax.set_xlabel('skew')
                ax.set_ylabel('$Q_{skew}$')
                # FIXME brier score for non-probabilistic outputs
                brier_score = normalized_brier_score(y==pos_label,prediction[:,pos_label])
                ax.set_title('{} BS = {}'.format(method, brier_score))

    df = df.convert_objects(convert_numeric=True)
    print df
    table =  df.pivot_table(values=['AUC'], index=['Experiment'],
                                  columns=['Method', 'Pos_label'])
    table.to_csv('table.csv', escape=False)
    # TODO consider the priors
    table_mean = df.pivot_table(values=['AUC'], index=['Experiment'],
                                  columns=['Method'], aggfunc=[np.mean])
    table_mean.to_csv('table_mean.csv', escape=False)

    print table_mean

    return 0


if __name__ == "__main__":
    sys.exit(main(pos_labels=[0], experiment_ids=[4]))
