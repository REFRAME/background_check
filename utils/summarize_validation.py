#!/usr/bin/python
from scipy import stats
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import itertools

# Not to crop the output columns
pd.set_option('expand_frame_repr', False)

class MyDataFrame(pd.DataFrame):
    def append_rows(self, rows):
        dfaux = pd.DataFrame(rows, columns=self.columns)
        return self.append(dfaux, ignore_index=True)

    def print_full(self):
        pd.set_option('display.max_rows', len(self))
        print(self)
        pd.reset_option('display.max_rows')

# TODO try to integrate into DataFrame when read from csv
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def table_number_iterations(df):
    methods = np.sort(df.method.unique())
    datasets = np.sort(df.dataset.unique())

    dfnit = MyDataFrame(columns=['dataset', 'method', 'iterations'])

    for dataset in datasets:
        dfaux = df[df['dataset']==dataset]
        for method in methods:
            n_iterations = dfaux[dfaux['method']==method].shape[0]
            dfnit = dfnit.append_rows([[dataset, method, n_iterations]])
    return dfnit


def wilcoxon_rank_sum_test_per_method_and_data(df, column='acc', signed=False,
                                               twosided=True):
    methods = np.sort(df.method.unique())
    datasets = np.sort(df.dataset.unique())

    dfstat = MyDataFrame(columns=['dataset', 'method1', 'method2', 'statistic',
                                  'pvalue'])

    for dataset in datasets:
        dfaux = df[df['dataset']==dataset]
        results = {}
        for method in methods:
            results[method] = dfaux[dfaux['method']==method][column]

        for (method1, method2) in itertools.combinations(methods,2):
            if signed:
                smallest = np.min([results[method1].shape[0],
                                   results[method2].shape[0]])
                statistic, pvalue = stats.wilcoxon(
                        results[method1][:smallest].values,
                        results[method2][:smallest].values)
            else:
                statistic, pvalue = stats.ranksums(results[method1].values,
                                                   results[method2].values)
            if not twosided:
                pvalue /= 2
            dfstat = dfstat.append_rows([[dataset, method1, method2, statistic,
                                          pvalue]])
    return dfstat


def wilcoxon_rank_sum_test_per_method(df, column='acc', signed=False,
                                      twosided=True):
    methods = np.sort(df.method.unique())
    datasets = np.sort(df.dataset.unique())
    results = {}
    for method in methods:
        results[method] = df[df['method']==method][column]

    dfstat = MyDataFrame(columns=['method1', 'method2', 'statistic', 'pvalue'])
    for (method1, method2) in itertools.combinations(methods,2):
        if method1 != method2:
            if signed:
                smallest = np.min([results[method1].shape[0],
                                   results[method2].shape[0]])
                statistic, pvalue = stats.wilcoxon(
                        results[method1][:smallest].values,
                        results[method2][:smallest].values)
            else:
                statistic, pvalue = stats.ranksums(results[method1].values,
                                                   results[method2].values)
            if not twosided:
                pvalue /= 2
            dfstat = dfstat.append_rows([[method1, method2, statistic,
                                          pvalue]])
    return dfstat


def friedman_test_per_method_and_data(df, column='acc'):
    methods = np.sort(df.method.unique())

    results = {}
    first = True
    for method in methods:
        results[method] = df[df['method']==method]['acc'].values
        if first:
            size = results[method].shape[0]
            first = False
        elif size != results[method].shape[0]:
            print("Friedman test can not be done with different sample sizes")
            return

    statistic, pvalue = stats.friedmanchisquare(results[methods[0]],
                                                results[methods[1]],
                                                results[methods[2]])
    from IPython import embed
    embed()
    return statistic, pvalue

def create_latex_table(table, acctest, logltest, measures=['acc', 'logloss'],
                       signed=False):
    methods = table.columns.levels[2].values
    datasets = np.sort(acctest.dataset.unique())

    if len(methods) > 2:
        print('LaTeX table not implemented for more than two methods yet')
        return

    header = ('\\begin{table}[t]\n'
              '\\centering\n'
              '\\scriptsize\n'
              '\\begin{tabular}{l|ll|ll}\n'
              '\\toprule')

    if signed:
        test = 'Wilcoxon signed-rank test'
    else:
        test = 'Wilcoxon rank-sum test'

    footer = ('\\bottomrule\n'
              '\\end{tabular}\n'
              '\\captionof{table}{Mean and standard deviation of the accuracy '
              'and log-loss on ' +str(len(datasets))+ ' datasets. '
              'A ' + test + ' has been performed per each metric '
              'and dataset comparing 20 iterations of 5-fold-crossvalidation'
              '; * significant at $p<0.05$; ** significant at $p<0.005$; '
              '*** significant at $p<0.001$.}\n'
              '\\label{tab:change:label}\n'
              '\\end{table}')
    print(header)

    line = ''
    for measure in measures:
        line += '\t&{}\t&'.format(measure)
    line += '\\\\'
    print(line)

    line = 'method'
    for measure in measures:
        for method in methods:
            line += '\t&{}'.format(method)
    line += '\\\\ \midrule'
    print(line)

    for dataset in datasets:
        line = dataset[:10]
        for measure in measures:
            if measure == 'acc':
                bold_value = table.loc[dataset, ('mean', measure)].values.max()
            elif measure == 'logloss':
                bold_value = table.loc[dataset, ('mean', measure)].values.min()

            for method in methods:
                bold = False
                line += '\t&\t'
                mean = table.loc[dataset, ('mean', measure, method)]
                std = table.loc[dataset, ('std', measure, method)]
                if measure == 'acc':
                    pvalue = acctest[acctest['dataset']==dataset]['pvalue'].values[0]
                elif measure == 'logloss':
                    pvalue = logltest[logltest['dataset']==dataset]['pvalue'].values[0]

                if mean == bold_value:
                    bold = True

                if measure == 'acc':
                    mean *= 100
                    std *= 100

                line += '$'
                if bold:
                    line += '\mathbf{'
                line += '{:2.2f}\pm{:.1f}'.format(mean,std)
                if bold:
                    if pvalue < 0.05:
                        line +='^{*'
                        if pvalue < 0.005:
                            line += '*'
                        if pvalue < 0.001:
                            line += '*'
                        line += '}'
                    line += '}'
                line += '$'

        line += '\\\\'
        print line
    print footer


def main(input_files, output_file):
    # if there is no column for logloss read_csv will create NaNs
    col_names = ['id1', 'id2', 'date', 'time', 'id_dat', 'dataset', 'id_met',
            'method', 'id_mc', 'mc', 'id_fol', 'test_fold', 'id_acc', 'acc',
            'id_log', 'logloss']
    first = True
    for input_file in input_files:
        print('Preprocessing {}'.format(input_file))
        if first:
            try:
                df = pd.read_csv(input_file, quotechar='|', header=None,
                        names=col_names)
                first = False

            except Exception as e:
                print e
                pass
        else:
            try:
                df2 = pd.read_csv(input_file, quotechar='|', header=None,
                        names=col_names)

                df = df.append(df2, ignore_index=True)

            except Exception as e:
                print e
                pass


    print("\nThe number of iterations per dataset and method\n")
    dfnit = table_number_iterations(df)

    table = dfnit.pivot_table(values=['iterations'], index=['dataset'],
                              columns=['method'])
    print_full(table)

    print("\nAccuracy and logloss per dataset and method\n")
    table = df.pivot_table(values=['acc', 'logloss'], index=['dataset'],
                           columns=['method'], aggfunc=[np.mean, np.std])
    table.to_csv(output_file)

    #table["accstd"] = table["acc"].map(str) + dataframe["quarter"]

    print_full(table)

    signeds = [True, False]

    methods = table.columns.levels[2].values

    for signed in signeds:
        test = 'Wilcoxon signed-rank' if signed else 'Wilcoxon rank-sum'
        print("\n{} test per dataset and method Accuracy\n".format(test))
        df_stat_acc = wilcoxon_rank_sum_test_per_method_and_data(df, column='acc',
                                                                 signed=signed,
                                                                 twosided=True)
        df_stat_acc.print_full()

        print("\n{} test per dataset and method Logloss\n".format(test))
        df_stat_logl = wilcoxon_rank_sum_test_per_method_and_data(df,
                column='logloss', signed=signed, twosided=True)
        df_stat_logl.print_full()

        create_latex_table(table, df_stat_acc, df_stat_logl, signed=signed)

        if len(methods) > 2:
            print("\nFriedman test per method and dataset\n")
            statistic, pvalue = friedman_test_per_method_and_data(df, column='acc')
            print("Statistic = {}, pvalue = {}".format(statistic, pvalue))

        print("\nAccuracy and logloss per method and all datasets\n")
        table_mean = df.pivot_table(values=['acc', 'logloss'], index=[],
                               columns=['method'], aggfunc=[np.mean, np.std])
        print_full(table_mean)

        print("\n{} test per method and all datasets Accuracy\n".format(test))
        dfstat = wilcoxon_rank_sum_test_per_method(df, column='acc',
                signed=signed, twosided=True)
        dfstat.print_full()

        print("\n{} test per method and all datasets Logloss\n".format(test))
        dfstat = wilcoxon_rank_sum_test_per_method(df, column='logloss',
                signed=signed, twosided=True)
        dfstat.print_full()


def parse_arguments():
    parser = ArgumentParser(description=('Reads all the validation files,'
                                         'merges them and generates a summary'))
    parser.add_argument('input_files', nargs='+',
                         help="A list of files to summarize")
    parser.add_argument('output_file',
                        help="A file to store the results for the output")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    main(args.input_files, args.output_file)
