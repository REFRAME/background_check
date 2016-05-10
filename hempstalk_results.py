from __future__ import division
import numpy as np
from sklearn.mixture import GMM
from sklearn.metrics import roc_auc_score
from scipy.stats import multivariate_normal
np.random.seed(42)
import matplotlib.pyplot as plt
plt.ion()
plt.rcParams['figure.figsize'] = (7,4)
plt.rcParams['figure.autolayout'] = True

from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from cwc.synthetic_data.datasets import MLData

import pandas as pd

mldata = MLData()

class MyMultivariateNormal(object):
    def __init__(self, min_covar=0.001, covariance_type='diag'):
        self.min_covar = min_covar
        self.covariance_type = covariance_type
        self.alpha = 0.00000001


        if covariance_type not in ['full', 'diag',]:
            raise ValueError('Invalid value for covariance_type: %s' %
                             covariance_type)

    def pseudo_determinant(self, A, alpha):
        n = len(A)
        return np.linalg.det(A + np.eye(n)*alpha)/ np.power(alpha, n-np.rank(A))

    def fit(self, x):
        self.mu = x.mean(axis=0)
        self.sigma = np.cov(x.T)
        self.sigma[self.sigma==0] = self.min_covar
        if(self.covariance_type == 'diag'):
            self.sigma = np.eye(np.alen(self.sigma))*self.sigma

        self.size = x.shape[1]

        self.det = np.linalg.det(self.sigma)
        # If sigma is singular
        if self.det == 0:
            self.pseudo_det = self.pseudo_determinant(self.sigma*2*np.pi, self.alpha)
            self.norm_const = 1.0/ np.sqrt(self.pseudo_det)
            self.inv = np.linalg.pinv(self.sigma)
        else:
            self.norm_const = 1.0/ ( np.power((2*np.pi),float(self.size)/2) *
                    np.sqrt(self.det) )
            self.inv = np.linalg.inv(self.sigma)

    def score(self,x):
        x_mu = np.subtract(x,self.mu)
        result = np.exp(-0.5 * np.diag(np.dot(x_mu,np.dot(self.inv,x_mu.T))))

        return self.norm_const * result


class MultivariateNormal(object):
    def __init__(self, allow_singular=True, covariance_type='diag'):
        self.allow_singular = allow_singular
        self.covariance_type = covariance_type

    def fit(self, x):
        self.mu = x.mean(axis=0)
        self.sigma = np.cov(x.T)
        if self.covariance_type == 'diag':
            self.sigma = np.eye(np.alen(self.sigma))*self.sigma

        self.model = multivariate_normal(mean=self.mu, cov=self.sigma,
                allow_singular=self.allow_singular)

    def score(self,x):
        return self.model.pdf(x)


def export_results(table):
    def mean_format(x):
        return "%1.3f" % x

    def std_format(x):
        return '$\pm$%1.2f' % x

    def dictionary_of_formats(table):
        functions = table.columns.levels[0]
        models = table.columns.levels[1]
        dict_format = {}
        for function in functions:
            for model in models:
                if function == 'mean':
                    dict_format[(function, model)] = mean_format
                elif function == 'std':
                    dict_format[(function, model)] = std_format
        return dict_format

    table.to_csv('final_table.csv')

    dict_format = dictionary_of_formats(table)
    table.to_latex('final_table.tex', formatters=dict_format, escape=False)


def separate_sets(x, y, test_fold_id, test_folds):
    x_test = x[test_folds == test_fold_id, :]
    y_test = y[test_folds == test_fold_id]

    x_train = x[test_folds != test_fold_id, :]
    y_train = y[test_folds != test_fold_id]
    return [x_train, y_train, x_test, y_test]


# Columns for the DataFrame
columns=['Dataset', 'MC iteration', 'N-fold id', 'Actual class', 'Model',
        'AUC', 'Prior']
# Create a DataFrame to record all intermediate results
df = pd.DataFrame(columns=columns)

mc_iterations = 10.0
n_folds = 10.0
for i, (name, dataset) in enumerate(mldata.datasets.iteritems()):
    print('Dataset number {}'.format(i))
    mldata.sumarize_datasets(name)
    for mc in np.arange(mc_iterations):
        skf = StratifiedKFold(dataset.target, n_folds=n_folds, shuffle=True)
        test_folds = skf.test_folds
        for test_fold in np.arange(n_folds):
            x_train, y_train, x_test, y_test = separate_sets(
                    dataset.data, dataset.target, test_fold, test_folds)
            n_training = np.alen(y_train)
            w_auc_fold_dens = 0
            w_auc_fold_bag = 0
            w_auc_fold_com = 0
            prior_sum = 0
            for actual_class in dataset.classes:
                tr_class = x_train[y_train == actual_class, :]
                t_labels = (y_test == actual_class).astype(int)
                prior = np.alen(tr_class) / n_training
                if np.alen(tr_class) > 1 and not all(t_labels == 0):
                    prior_sum += prior
                    n_c = tr_class.shape[1]
                    if n_c > np.alen(tr_class):
                        n_c = np.alen(tr_class)


                    # Train a Density estimator
                    #model = GMM(n_components=n_c,
                    #            covariance_type='full')
                    model = GMM(n_components=1, covariance_type='diag')
                    #model = MyMultivariateNormal(covariance_type='full')
                    #model = MultivariateNormal(covariance_type='diag')

                    model.fit(tr_class)

                    new_data = model.sample(np.alen(tr_class))

                    # Train a Bag of Trees
                    bag = BaggingClassifier(
                        base_estimator=DecisionTreeClassifier(),
                        n_estimators=10)

                    new_data = np.vstack((tr_class, new_data))
                    y = np.zeros(np.alen(new_data))
                    y[:np.alen(tr_class)] = 1

                    bag.fit(new_data, y)

                    # Combine the results
                    probs = bag.predict_proba(x_test)[:, 1]
                    scores = model.score(x_test)

                    com_scores = (probs / np.clip(1.0 - probs,
                                                  1e-16, 1.0)) * np.exp(scores)

                    # Scores for the Density estimator
                    auc_dens = roc_auc_score(t_labels, scores)
                    # Scores for the Bag of trees
                    auc_bag = roc_auc_score(t_labels, probs)
                    # Scores for the Combined model
                    auc_com = roc_auc_score(t_labels, com_scores)

                    # Create a new DataFrame to append to the original one
                    df2 = pd.DataFrame([[name, mc, test_fold, actual_class,
                                         'Combined', auc_com, prior],
                                        [name, mc, test_fold, actual_class,
                                         'P(T$|$X)', auc_bag, prior],
                                        [name, mc, test_fold, actual_class,
                                         'P(X$|$A)', auc_dens, prior]],
                                        columns=columns)
                    df = df.append(df2, ignore_index=True)


# Convert values to numeric
df = df.convert_objects(convert_numeric=True)

# Group everything except classes
dfgroup_classes = df.groupby(by=['Dataset', 'MC iteration', 'N-fold id',
                                 'Model'])
# Compute the Prior sum for each dataset, iteration and fold
df['Prior_sum'] = dfgroup_classes['Prior'].transform(np.sum)
# Compute the individual weighted AUC per each class and experiment
df['wAUC'] = df.Prior * df.AUC / df.Prior_sum

# Sum the weighted AUC of each class per each experiment
series_wAUC = dfgroup_classes['wAUC'].sum()

# Transform the series to a DataFrame
df_wAUC = series_wAUC.reset_index(inplace=False)
# Compute mean and standard deviation of wAUC per Dataset and model
final_results = df_wAUC.groupby(['Dataset', 'Model'])['wAUC'].agg([np.mean,
    np.std])
# Transform the series to a DataFrame
final_results.reset_index(inplace=True)

# Represent the results in a table format
final_table =  final_results.pivot_table(values=['mean', 'std'],
                                         index=['Dataset'], columns=['Model'])

# Export the results in a csv and LaTeX file
export_results(final_table)
