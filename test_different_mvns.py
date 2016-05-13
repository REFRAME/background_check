from __future__ import division
import numpy as np
from sklearn.mixture import GMM
from sklearn.metrics import roc_auc_score
np.random.seed(42)
import matplotlib.pyplot as plt
plt.ion()
plt.rcParams['figure.figsize'] = (7,4)
plt.rcParams['figure.autolayout'] = True

from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from cwc.synthetic_data.datasets import MLData
from cwc.models.density_estimators import MultivariateNormal
from cwc.models.density_estimators import MyMultivariateNormal

import pandas as pd

mldata = MLData()

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


def mean_squared_error(x1, x2):
    return np.mean(np.power(np.subtract(x1, x2),2))


# Columns for the DataFrame to compare the Density estimators
columns_gaussians=['Dataset', 'MC iteration', 'N-fold id', 'Actual class',
                   'Model', 'mean_mse','cov_mse', 'Prior']
# Create a DataFrame to record all intermediate results
df_gaussians = pd.DataFrame(columns=columns_gaussians)

# Columns for the DataFrame
columns=['Dataset', 'MC iteration', 'N-fold id', 'Actual class', 'Model',
        'AUC', 'Prior']
# Create a DataFrame to record all intermediate results
df = pd.DataFrame(columns=columns)

mc_iterations = 10
n_folds = 10
for i, (name, dataset) in enumerate(mldata.datasets.iteritems()):
    print('Dataset number {}'.format(i))
    if name == 'MNIST':
        # TODO get a stratified portion of the validation set [0:60000]
        dataset._data = dataset._data[-10000:]
        dataset._target = dataset._target[-10000:]

    mldata.sumarize_datasets(name)
    for mc in np.arange(mc_iterations):
        skf = StratifiedKFold(dataset.target, n_folds=n_folds, shuffle=True)
        test_folds = skf.test_folds
        for test_fold in np.arange(n_folds):
            x_train, y_train, x_test, y_test = separate_sets(
                    dataset.data, dataset.target, test_fold, test_folds)
            n_training = np.alen(y_train)
            for actual_class in dataset.classes:
                tr_class = x_train[y_train == actual_class, :]
                t_labels = (y_test == actual_class).astype(int)
                prior = np.alen(tr_class) / n_training
                if np.alen(tr_class) > 1 and not all(t_labels == 0):
                    n_c = tr_class.shape[1]
                    if n_c > np.alen(tr_class):
                        n_c = np.alen(tr_class)


                    # Train a Density estimator
                    model_mymvn = MyMultivariateNormal(covariance_type='diag')
                    model_mymvn.fit(tr_class)

                    model_mvn= MultivariateNormal(covariance_type='diag')
                    model_mvn.fit(tr_class)

                    model_gmm = GMM(n_components=1, covariance_type='diag')
                    model_gmm.fit(tr_class)

                    x_train_means = x_train.mean(axis=0)
                    x_train_stds = x_train.std(axis=0)

                    mse_mean_mymvn = mean_squared_error(model_mymvn.means_,
                                                  x_train_means)
                    mse_std_mymvn = mean_squared_error(model_mymvn.covars_,
                                                  np.power(x_train_stds,2))
                    mse_mean_mvn = mean_squared_error(model_mvn.means_,
                                                  x_train_means)
                    mse_std_mvn = mean_squared_error(model_mvn.covars_,
                                                  np.power(x_train_stds,2))
                    mse_mean_gmm = mean_squared_error(model_gmm.means_,
                                                  x_train_means)
                    mse_std_gmm = mean_squared_error(model_gmm.covars_,
                                                  np.power(x_train_stds,2))
                    dfaux = pd.DataFrame([[name, mc, test_fold, actual_class,
                                         'MyMVN', mse_mean_mymvn, mse_std_mymvn, prior],
                                        [name, mc, test_fold, actual_class,
                                         'MVN', mse_mean_mvn, mse_std_mvn, prior],
                                        [name, mc, test_fold, actual_class,
                                         'GMM', mse_mean_gmm, mse_std_gmm, prior]],
                                        columns=columns_gaussians)
                    df_gaussians = df_gaussians.append(dfaux, ignore_index=True)

                    # Scores for the Density estimator
                    auc_dens = roc_auc_score(t_labels, model_gmm.score(x_test))
                    # Scores for the Bag of trees
                    auc_mvn = roc_auc_score(t_labels, model_mvn.score(x_test))
                    # Scores for the Density estimator
                    auc_mymvn = roc_auc_score(t_labels,
                            model_mymvn.log_likelihood(x_test))

                    # Create a new DataFrame to append to the original one
                    dfaux = pd.DataFrame([[name, mc, test_fold, actual_class,
                                         'GMM', auc_dens, prior],
                                        [name, mc, test_fold, actual_class,
                                         'MyMVN', auc_mymvn, prior],
                                        [name, mc, test_fold, actual_class,
                                         'MVN', auc_mvn, prior]],
                                        columns=columns)
                    df = df.append(dfaux, ignore_index=True)


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

df_gaus_group = df_gaussians.groupby(['Dataset', 'Model']).mean()
df_gaus_group.reset_index(inplace=True)
gaus_table = df_gaus_group.pivot_table(values=['mean_mse', 'cov_mse'],
                                       index=['Dataset'], columns=['Model'])
gaus_table.to_csv('gaus_table.csv')
