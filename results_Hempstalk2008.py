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

from sklearn.metrics import roc_curve
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
from cwc.data_wrappers.datasets import Data
from cwc.data_wrappers import reject
from cwc.visualization.cost_lines import plot_skew_lines
from cwc.visualization.roc_analysis import plot_roc_curve
from cwc.models.background_check import BackgroundCheck
from diary import Diary

import pandas as pd

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
    print table
    table.to_latex('final_table.tex', formatters=dict_format, escape=False)


def separate_sets(x, y, test_fold_id, test_folds):
    x_test = x[test_folds == test_fold_id, :]
    y_test = y[test_folds == test_fold_id]

    x_train = x[test_folds != test_fold_id, :]
    y_train = y[test_folds != test_fold_id]
    return [x_train, y_train, x_test, y_test]


def mean_squared_error(x1, x2):
    return np.mean(np.power(np.subtract(x1, x2),2))

def generate_and_save_plots(labels, scores, diary, name, mc, test_fold,
                            actual_class, method):
    fig = plt.figure('roc_curve')
    title = '{}_{}_{}_{}_{}'.format(name,mc,test_fold,actual_class,method)
    plot_roc_curve(t_labels,scores,pos_label=1, fig=fig,
                   title=title)
    diary.save_figure(fig,'{}_roc_curve'.format(title))

    fig = plt.figure('skew_lines')
    plot_skew_lines(t_labels,scores,pos_label=1,
            lower_envelope=True, fig=fig, title=title)
    diary.save_figure(fig,'{}_skew_lines'.format(title))


def main():
    dataset_names = ['diabetes', 'ecoli', 'glass', 'heart-statlog',
                     'ionosphere', 'iris', 'letter', 'mfeat-karhunen',
                     'mfeat-morphological', 'mfeat-zernike', 'optdigits',
                     'pendigits', 'sonar', 'vehicle', 'waveform-5000']

    data = Data(dataset_names=dataset_names)

    diary = Diary(name='hempstalk', path='results', overwrite=False,
                  fig_format='svg')
    diary.add_notebook('cross_validation')

    # Columns for the DataFrame
    columns=['Dataset', 'MC iteration', 'N-fold id', 'Actual class', 'Model',
            'AUC', 'Prior']
    # Create a DataFrame to record all intermediate results
    df = pd.DataFrame(columns=columns)

    mc_iterations = 10
    n_folds = 10

    gammas = {"diabetes":0.00005, "ecoli":0.1, "glass":0.005,
              "heart-statlog":0.0001, "ionosphere":0.00005, "iris":0.0005,
              "letter":0.000005, "mfeat-karhunen":0.0001,
              "mfeat-morphological":0.0000001, "mfeat-zernike":0.000001,
              "optdigits":0.00005, "pendigits":0.000001, "sonar":0.001,
              "vehicle":0.00005, "waveform-5000":0.001}

    for i, (name, dataset) in enumerate(data.datasets.iteritems()):
        print('Dataset number {}'.format(i))

        data.sumarize_datasets(name)
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
                    tr_class_unique_values = [np.unique(tr_class[:,column]).shape[0] for column in
                                              range(tr_class.shape[1])]
                    cols_keep = np.where(np.not_equal(tr_class_unique_values,1))[0]
                    tr_class = tr_class[:,cols_keep]
                    x_test_cleaned = x_test[:,cols_keep]
                    t_labels = (y_test == actual_class).astype(int)
                    prior = np.alen(tr_class) / n_training
                    if np.alen(tr_class) > 1 and not all(t_labels == 0):
                        prior_sum += prior
                        n_c = tr_class.shape[1]
                        if n_c > np.alen(tr_class):
                            n_c = np.alen(tr_class)


                        # Train a Density estimator
                        model_gmm = GMM(n_components=1, covariance_type='diag')
                        model_gmm.fit(tr_class)

                        sv = OneClassSVM(nu=0.1, gamma=0.5)
                        bc = BackgroundCheck(estimator=sv)
                        bc.fit(tr_class)
                        svm_scores = bc.predict_proba(x_test_cleaned)[:, 1]
                        # Generate artificial data
                        new_data = model_gmm.sample(np.alen(tr_class))

                        # Train a Bag of Trees
                        bag = BaggingClassifier(
                            base_estimator=DecisionTreeClassifier(),
                            n_estimators=10)

                        new_data = np.vstack((tr_class, new_data))
                        y = np.zeros(np.alen(new_data))
                        y[:np.alen(tr_class)] = 1

                        bag.fit(new_data, y)

                        # Combine the results
                        probs = bag.predict_proba(x_test_cleaned)[:, 1]
                        scores = model_gmm.score(x_test_cleaned)

                        com_scores = (probs / np.clip(1.0 - probs, np.float32(1e-32), 1.0)) * (scores-scores.min())


                        # Generate our new data
                        # FIXME solve problem with #samples < #features
                        pca=True
                        if tr_class.shape[0] < tr_class.shape[1]:
                            pca=False
                        our_new_data = reject.create_reject_data(
                                            tr_class, proportion=1,
                                            method='uniform_hsphere', pca=pca,
                                            pca_variance=0.99, pca_components=0,
                                            hshape_cov=0, hshape_prop_in=0.99,
                                            hshape_multiplier=1.5)
                        our_new_data = np.vstack((tr_class, our_new_data))
                        y = np.zeros(np.alen(our_new_data))
                        y[:np.alen(tr_class)] = 1

                        # Train Our Bag of Trees
                        our_bag = BaggingClassifier(
                            base_estimator=DecisionTreeClassifier(),
                            n_estimators=10)
                        our_bag.fit(our_new_data, y)
                        # Combine the results
                        our_probs = our_bag.predict_proba(x_test_cleaned)[:, 1]

                        our_comb_scores = (our_probs / np.clip(1.0 - our_probs,
                                np.float32(1e-32), 1.0)) * (scores-scores.min())

                        # Scores for the Density estimator
                        auc_dens = roc_auc_score(t_labels, scores)
                        # Scores for the Bag of trees
                        auc_bag = roc_auc_score(t_labels, probs)
                        # Scores for the Combined model
                        auc_com = roc_auc_score(t_labels, com_scores)
                        # Scores for our Bag of trees (trained on our data)
                        auc_our_bag = roc_auc_score(t_labels, our_probs)
                        # Scores for our Bag of trees (trained on our data)
                        auc_our_comb = roc_auc_score(t_labels, our_comb_scores)
                        # Scores for the Background Check with SVm
                        auc_svm = roc_auc_score(t_labels, svm_scores)

                        # Create a new DataFrame to append to the original one
                        dfaux = pd.DataFrame([[name, mc, test_fold, actual_class,
                                             'Combined', auc_com, prior],
                                            [name, mc, test_fold, actual_class,
                                             'P(T$|$X)', auc_bag, prior],
                                            [name, mc, test_fold, actual_class,
                                             'P(X$|$A)', auc_dens, prior],
                                            [name, mc, test_fold, actual_class,
                                             'Our Bagg', auc_our_bag, prior],
                                            [name, mc, test_fold, actual_class,
                                             'Our Combined', auc_our_comb, prior],
                                            [name, mc, test_fold, actual_class,
                                             'SVM_BC', auc_svm, prior]],
                                             columns=columns)
                        df = df.append(dfaux, ignore_index=True)

                        # generate_and_save_plots(t_labels, scores, diary, name, mc, test_fold,
                        #                         actual_class, 'P(X$|$A)')
                        # generate_and_save_plots(t_labels, probs, diary, name, mc, test_fold,
                        #                         actual_class, 'P(T$|$X)')
                        # generate_and_save_plots(t_labels, com_scores, diary, name, mc, test_fold,
                        #                         actual_class, 'Combined')
                        # generate_and_save_plots(t_labels, our_probs, diary, name, mc, test_fold,
                        #                         actual_class, 'Our_Bagg')
                        # generate_and_save_plots(t_labels, our_comb_scores, diary, name, mc, test_fold,
                        #                         actual_class, 'Our_Combined')
                        # generate_and_save_plots(t_labels, svm_scores, diary,
                        #                         name, mc, test_fold,
                        #                         actual_class, 'SVM_BC')



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


if __name__=='__main__':
    main()
