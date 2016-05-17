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

colors = ['red', 'blue', 'magenta']

def one_vs_rest_roc_curve(y,p, pos_label=0):
    """Returns the roc curve of class 0 vs the rest of the classes"""
    aux = np.copy(y)
    aux[aux!=0] = 1
    return roc_curve(aux, p, pos_label=0)

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


def plot_roc_curve(y,p,fig=None,title=''):
    if fig is None:
        fig = plt.figure('roc_curve')
    fig.clf()

    roc = one_vs_rest_roc_curve(y, p)
    auroc = auc(roc[0], roc[1])
    ax = fig.add_subplot(111)
    ax.plot(roc[0], roc[1])
    ax.plot([0,1],[0,1], 'g--')
    ax.plot([0,1],[1,0], 'g--')
    upper_hull = convex_hull(zip(roc[0],roc[1]))
    rg_hull, pg_hull = zip(*upper_hull)
    plt.plot(rg_hull, pg_hull, 'r--')
    ax.set_title('{0} {1:.3f}'.format(title, auroc))

    return auroc

class BackgroundClass(object):
    def __init__(self, i=1):
        self.samples, self.mean, self.cov = self.background_description(i)
        self.mvn = MyMultivariateNormal(self.mean, self.cov)

    def set_experiment(self, i):
        self.__init__(i)

    def sample(self, n_samples=None):
        if n_samples is not None:
            return self.mvn.sample(n_samples)
        return self.mvn.sample(self.samples)

    @property
    def n_experiments(self):
        return 8

    @property
    def experiment_ids(self):
        return range(self.n_experiments)

    def background_description(self, i, samples=500):
        if i==0:
            samples = 0
            mean = np.array([0,0])
            cov = np.array([[1,0], [0,1]])
        elif i==1:
            samples = samples
            mean = np.array([0,0])
            cov = np.array([[1,0], [0,1]])
        elif i==2:
            samples = samples
            mean = np.array([1,1])
            cov = np.array([[1,0], [0,1]])
        elif i==3:
            samples = samples
            mean = np.array([2,2])
            cov = np.array([[1,0], [0,1]])
        elif i==4:
            samples = samples
            mean = np.array([-3,3])
            cov = np.array([[1,0], [0,1]])
        elif i==5:
            samples = samples
            mean = np.array([-2,4])
            cov = np.array([[1,0], [0,1]])
        elif i==6:
            samples = samples
            mean = np.array([-1,5])
            cov = np.array([[1,0], [0,1]])
        elif i==7:
            samples = samples
            mean = np.array([0,6])
            cov = np.array([[1,0], [0,1]])
        else:
            raise Exception('Unknown experiment')

        return samples, mean, cov

class MyDataFrame(pd.DataFrame):
    def append_rows(self, rows):
        dfaux = pd.DataFrame(rows, columns=self.columns)
        return self.append(dfaux, ignore_index=True)

def main():
    np.random.seed(42)

    # Columns for the DataFrame
    columns=['Experiment', 'Method', 'AUC']
    # Create a DataFrame to record all intermediate results
    df = MyDataFrame(columns=columns)

    # Two original classes
    samples = np.array([500,         # Class 0
                        500])         # Class 1
    means = np.array([[0,0],       # Class 1
                      [2,2]])       # Class 2
    covs = np.array([[[1,0],       # Class 1
                      [0,1]],
                     [[1,0],       # Class 2
                      [0,1]]])

    mvn0 = MyMultivariateNormal(means[0], covs[0])
    mvn1 = MyMultivariateNormal(means[1], covs[1])
    x0 = mvn0.sample(samples[0])
    x1 = mvn1.sample(samples[1])

    bg_class = BackgroundClass()
    for bg_class_id in bg_class.experiment_ids:
        # Background
        bg_class.set_experiment(bg_class_id)
        xb = bg_class.sample()

        x = np.vstack([x0, x1, xb])
        y = np.hstack([np.zeros(len(x0)), np.ones(len(x1)),
            np.ones(len(xb))*2]).astype(int)

        x_grid = get_grid(x)

        p0_grid =  mvn0.score(x_grid)
        p1_grid =  mvn1.score(x_grid)

        prior = samples/sum(samples)
        posterior = (p0_grid*prior[0])/(p0_grid*prior[0]+p1_grid*prior[1])

        fig = plt.figure('Density')
        plot_data_and_contourlines(x,y,x_grid,[p0_grid,p1_grid],delta=50,fig=fig, title='Density')

        fig = plt.figure('Bayes')
        plot_data_and_contourlines(x,y,x_grid,[posterior],delta=50, fig=fig, title='Bayes optimal')

        q0 = mvn0.score(x)
        fig = plt.figure('roc_density_0')
        auroc = plot_roc_curve(y,q0, fig=fig, title='ROC Density class 0')
        df = df.append_rows([[bg_class_id, 'Density_0', auroc]])

        fig = plt.figure('P_Miquel')
        plot_data_and_contourlines(x,y,x_grid,[p0_grid*posterior,p1_grid*(1-posterior)]
                                   ,delta=50,fig=fig, title='P_Miquel')
        q0 = mvn0.score(x)
        q1 = mvn1.score(x)
        posterior = ((q0**2)*prior[0])/(q0*prior[0]+q1*prior[1])
        fig = plt.figure('roc_miquel_posterior')
        auroc = plot_roc_curve(y, posterior, fig=fig, title='ROC Miquel')
        df = df.append_rows([[bg_class_id, 'Miquel', auroc]])


        q0 = mvn0.score(x)
        q1 = mvn1.score(x)
        q_posterior = (q0*prior[0])/(q0*prior[0]+q1*prior[1])
        fig = plt.figure('roc_curve_posterior')
        auroc = plot_roc_curve(y, q_posterior, fig=fig, title='ROC Bayes Optimal')
        df = df.append_rows([[bg_class_id, 'Bayes_Optimal', auroc]])


        P_x_t = p0_grid*prior[0]+p1_grid*prior[1]
        P_t = 0.8 # np.in1d(y,[0,1]).sum()/len(y)
        P_b = (1-P_t)
        max_value = np.maximum(mvn0.score([means[0]])*prior[0] + mvn1.score([means[0]])*prior[0],
                               mvn0.score([means[1]])*prior[1] + mvn1.score([means[1]])*prior[1])
        P_x_b = max_value-P_x_t

        numerator = p0_grid*prior[0]*P_t
        denominator = numerator + p1_grid*prior[1]*P_t + P_x_b*P_b
        P_t_0_x = numerator/denominator

        numerator = p1_grid*prior[1]*P_t
        denominator = numerator + p0_grid*prior[0]*P_t + P_x_b*P_b
        P_t_1_x = numerator/denominator
        fig = plt.figure('P_telmo')
        plot_data_and_contourlines(x,y,x_grid,[P_t_0_x,P_t_1_x],delta=50, fig=fig, title='P_telmo')


        # Compute predictions for all samples
        P_x_t = q0*prior[0]+q1*prior[1]
        P_x_b = max_value-P_x_t
        numerator = q0*prior[0]*P_t
        denominator = numerator + q1*prior[1]*P_t + P_x_b*P_b
        P_t_0_x = numerator/denominator

        fig = plt.figure('roc_telmo_posterior')
        auroc = plot_roc_curve(y, P_t_0_x, fig=fig, title='ROC Telmo')
        df = df.append_rows([[bg_class_id, 'Telmo', auroc]])


        # SVC
        x0_train = mvn0.sample(samples[0])
        x1_train = mvn1.sample(samples[1])

        x_train = np.vstack([x0_train, x1_train])
        y_train = np.hstack([np.zeros(len(x0_train)), np.ones(len(x1_train))]).astype(int)

        svc = svm.SVC(probability=True)
        svc.fit(x_train,y_train)
        svc_pred = svc.predict_proba(x_grid)
        fig = plt.figure('svm')
        plot_data_and_contourlines(x,y,x_grid,[svc_pred[:,0],svc_pred[:,1]],delta=50,
                fig=fig, title='P_SVC')

        predictions = svc.predict_proba(x)
        fig = plt.figure('roc_svc')
        auroc = plot_roc_curve(y, predictions[:,0], fig=fig, title='ROC SVC')
        df = df.append_rows([[bg_class_id, 'SVC', auroc]])

    df = df.convert_objects(convert_numeric=True)
    final_table =  df.pivot_table(values=['AUC'], index=['Experiment'],
                                  columns=['Method'])
    print final_table

    return 0

if __name__ == "__main__":
    sys.exit(main())
