import warnings
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from math import ceil
from math import sqrt
import numpy as np
plt.ion()

class Dataset(object):
    def __init__(self, name, data, target):
        self.name = name
        self._data = self.standardize_data(data)
        self._target, self._classes, self._names, self._counts = self.standardize_targets(target)

    def standardize_data(self, data):
        new_data = data.astype(float)
        data_mean = new_data.mean(axis=0)
        data_std = new_data.std(axis=0)
        data_std[data_std == 0] = 1
        return (new_data-data_mean)/data_std

    def standardize_targets(self, target):
        target = np.squeeze(target)
        names, counts = np.unique(target, return_counts=True)
        new_target = np.empty_like(target, dtype=int)
        for i, name in enumerate(names):
            new_target[target==name] = i
        classes = range(len(names))
        return new_target, classes, names, counts

    @property
    def target(self):
        return self._target

    #@target.setter
    #def target(self, new_value):
    #    self._target = new_value

    @property
    def data(self):
        return self._data

    @property
    def names(self):
        return self._names

    @property
    def classes(self):
        return self._classes

    @property
    def counts(self):
        return self._counts

    def print_summary(self):
        print('Name = {}'.format(self.name))
        print('Data shape = {}'.format(self.data.shape))
        print('Target shape = {}'.format(self.target.shape))
        print('Target classes = {}'.format(self.classes))
        print('Target labels = {}'.format(self.names))


class MLData(object):
    mldata_names = {'diabetes':'diabetes',
                    'ecoli':'uci-20070111 ecoli',
                    'glass':'glass',
                    'heart-statlog':'datasets-UCI heart-statlog',
                    'ionosphere':'ionosphere',
                    'iris':'iris',
                    'letter':'letter',
                    'mfeat-karhunen':'uci-20070111 mfeat-karhunen',
                    'mfeat-morphological':'uci-20070111 mfeat-morphological',
                    'mfeat-zernike':'uci-20070111 mfeat-zernike',
                    'optdigits':'uci-20070111 optdigits',
                    'pendigits':'uci-20070111 pendigits',
                    'sonar':'sonar',
                    'vehicle':'vehicle',
                    'waveform-5000':'datasets-UCI waveform-5000',
                    'scene-classification':'scene-classification',
                    #'spam':'uci-20070111 spambase',
                    'tic-tac':'uci-20070111 tic-tac-toe',
                    'MNIST':'MNIST (original)'}

    def __init__(self, data_home='./datasets/', load_all=False):
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(('This Class is going to be deprecated in a future '
                       'version, please use cwc.synthetic_data.Data instead.'),
                      DeprecationWarning)
        self.data_home = data_home
        self.datasets = {}

        if load_all:
            for key in MLData.mldata_names.keys():
                self.datasets[key] = self.get_dataset(key)

    def get_dataset(self, name):
        mldata = fetch_mldata(MLData.mldata_names[name], data_home=self.data_home)

        if name=='ecoli':
            data = mldata.target.T
            target = mldata.data
        elif name=='diabetes':
            data = mldata.data
            target = mldata.target
        elif name=='optdigits':
            data = mldata.data[:,:-1]
            target = mldata.data[:,-1]
        elif name=='pendigits':
            data = mldata.data[:,:-1]
            target = mldata.data[:,-1]
        elif name=='waveform-5000':
            data = mldata.target.T
            target = mldata.data
        elif name=='heart-statlog':
            data = np.hstack([mldata['target'].T, mldata.data, mldata['int2'].T])
            target = mldata['class']
        elif name=='mfeat-karhunen':
            data = mldata.target.T
            target = mldata.data
        elif name=='mfeat-zernike':
            data = mldata.target.T
            target = mldata.data
        elif name=='mfeat-morphological':
            data = mldata.target.T
            target = mldata.data
        elif name=='scene-classification':
            data = mldata.data
            target = mldata.target.toarray()
            target = target.transpose()[:,4]
        elif name=='tic-tac':
            n = np.alen(mldata.data)
            data = np.hstack((mldata.data.reshape(n,1),
                                     np.vstack([mldata[feature] for feature in
                                                mldata.keys() if 'square' in
                                                feature]).T,
                                     mldata.target.reshape(n,1)))
            for i, value in enumerate(np.unique(data)):
                data[data==value] = i
            data = data.astype(float)
            target = mldata.Class.reshape(n,1)
        else:
            try:
                data = mldata.data
                target = mldata.target
            except KeyError:
                print "KeyError: {}".format(name)
                raise
            except AttributeError:
                print "AttributeError: {}".format(name)
                raise

        return Dataset(name, data, target)


    def sumarize_datasets(self, name=None):
        if name is not None:
            dataset = self.datasets[name]
            dataset.print_summary()
        else:
            for name, dataset in self.datasets.iteritems():
                dataset.print_summary()

class Data(object):
    mldata_names = {'diabetes':'diabetes',
                    'ecoli':'uci-20070111 ecoli',
                    'glass':'glass',
                    'heart-statlog':'datasets-UCI heart-statlog',
                    'ionosphere':'ionosphere',
                    'iris':'iris',
                    'letter':'letter',
                    'mfeat-karhunen':'uci-20070111 mfeat-karhunen',
                    'mfeat-morphological':'uci-20070111 mfeat-morphological',
                    'mfeat-zernike':'uci-20070111 mfeat-zernike',
                    'optdigits':'uci-20070111 optdigits',
                    'pendigits':'uci-20070111 pendigits',
                    'sonar':'sonar',
                    'vehicle':'vehicle',
                    'waveform-5000':'datasets-UCI waveform-5000',
                    'scene-classification':'scene-classification',
                    #'spam':'uci-20070111 spambase', # not working
                    'tic-tac':'uci-20070111 tic-tac-toe',
                    'MNIST':'MNIST (original)',
                    'autos':'uci-20070111 autos',
                    'car':'uci-20070111 car',
                    'cleveland':'uci-20070111 cleveland',
                    'dermatology':'uci-20070111 dermatology',
                    'flare':'uci-20070111 solar-flare_2',
                    'page-blocks':'uci-20070111 page-blocks',
                    'segment':'datasets-UCI segment',
                    'shuttle':'shuttle',
                    'vowel':'uci-20070111 vowel',
                    'zoo':'uci-20070111 zoo',
                    'abalone':'uci-20070111 abalone',
                    'balance-scale': 'uci-20070111 balance-scale',
                    # To be added:
                    'credit-approval':'uci-20070111 credit-a',
                    # Need preprocessing :
                    'auslan':'',
                    # Needs to be generated
                    'led7digit':'',
                    'yeast':'',
                    # Needs permission from ml-repository@ics.uci.edu
                    'lymphography':'',
                    # HTTP Error 500 in mldata.org
                    'satimage':'satimage',
                    'nursery':'uci-20070111 nursery'
                    }

    def __init__(self, data_home='./datasets/', dataset_names=None, load_all=False):
        self.data_home = data_home
        self.datasets = {}

        if load_all:
            dataset_names = Data.mldata_names.keys()
            self.load_datasets_by_name(dataset_names)
        elif dataset_names is not None:
            self.load_datasets_by_name(dataset_names)


    def load_datasets_by_name(self, names):
        for name in names:
            self.datasets[name] = self.get_dataset_by_name(name)


    def get_dataset_by_name(self, name):
        if name in Data.mldata_names.keys():
            return self.get_mldata_dataset(name)
        elif name == 'spambase':
            data = np.load('./datasets/uci_spambase.npy')
            target = data[:,-1]
            data = data[:,0:-1]
        else:
            raise Exception('Dataset {} not available'.format(name))
        return Dataset(name, data, target)


    def get_mldata_dataset(self, name):
        mldata = fetch_mldata(Data.mldata_names[name], data_home=self.data_home)

        if name=='ecoli':
            data = mldata.target.T
            target = mldata.data
        elif name=='diabetes':
            data = mldata.data
            target = mldata.target
        elif name=='optdigits':
            data = mldata.data[:,:-1]
            target = mldata.data[:,-1]
        elif name=='pendigits':
            data = mldata.data[:,:-1]
            target = mldata.data[:,-1]
        elif name=='waveform-5000':
            data = mldata.target.T
            target = mldata.data
        elif name=='heart-statlog':
            data = np.hstack([mldata['target'].T, mldata.data, mldata['int2'].T])
            target = mldata['class']
        elif name=='mfeat-karhunen':
            data = mldata.target.T
            target = mldata.data
        elif name=='mfeat-zernike':
            data = mldata.target.T
            target = mldata.data
        elif name=='mfeat-morphological':
            data = mldata.target.T
            target = mldata.data
        elif name=='scene-classification':
            data = mldata.data
            target = mldata.target.toarray()
            target = target.transpose()[:,4]
        elif name=='tic-tac':
            n = np.alen(mldata.data)
            data = np.hstack((mldata.data.reshape(n,1),
                                     np.vstack([mldata[feature] for feature in
                                                mldata.keys() if 'square' in
                                                feature]).T,
                                     mldata.target.reshape(n,1)))
            for i, value in enumerate(np.unique(data)):
                data[data==value] = i
            data = data.astype(float)
            target = mldata.Class.reshape(n,1)
        elif name=='autos':
            target = mldata.int5[5,:].reshape(-1,1)
            data = np.hstack((
                      mldata['target'].reshape(-1,1),
                      self.nominal_to_float(mldata['data'].reshape(-1,1)),
                      self.nominal_to_float(mldata['fuel-type'].reshape(-1,1)),
                      self.nominal_to_float(mldata['aspiration'].reshape(-1,1)),
                      self.nominal_to_float(mldata['num-of-doors'].reshape(-1,1)),
                      self.nominal_to_float(mldata['body-style'].reshape(-1,1)),
                      self.nominal_to_float(mldata['drive-wheels'].reshape(-1,1)),
                      self.nominal_to_float(mldata['engine-location'].reshape(-1,1)),
                      mldata['double1'].T.reshape(-1,4),
                      mldata['int2'].reshape(-1,1),
                      self.nominal_to_float(mldata['engine-type'].reshape(-1,1)),
                      self.nominal_to_float(mldata['num-of-cylinders'].reshape(-1,1)),
                      mldata['int3'].reshape(-1,1),
                      self.nominal_to_float(mldata['fuel-system'].reshape(-1,1)),
                      mldata['double4'].T.reshape(-1,3),
                      mldata['int5'][:-1,:].T.reshape(-1,5)
                              ))
            missing = np.logical_or(np.isnan(data),
                                    data == -2147483648).any(axis=1)
            data = data[~missing]
            target = target[~missing]
        elif name=='car':
            target = mldata['class']
            feature_names = ['data', 'target', 'doors', 'persons', 'lug_boot',
                             'safety']
            data = np.hstack([
                    self.nominal_to_float(mldata[f_name].reshape(-1,1))
                        for f_name in feature_names])
        elif name=='cleveland':
            target = mldata.int2.reshape(-1,1)
            data = np.hstack((mldata.target.T, mldata.data))
            missing = np.logical_or(np.isnan(data),
                                    data == -2147483648).any(axis=1)
            data = data[~missing]
            target = target[~missing]
        elif name=='dermatology':
            target = mldata.data[:,0]
            data = mldata.data[:,1:]
            missing = np.logical_or(np.isnan(data),
                                    data == -2147483648).any(axis=1)
            data = data[~missing]
            target = target[~missing]
        elif name=='flare':
            target = mldata.target
            data = mldata['int0'].T

            # TODO this dataset is divided in two files, see more elegant way
            # to add it
            mldata = fetch_mldata('uci-20070111 solar-flare_1')

            target = np.hstack((target, mldata.target))
            data = np.vstack((data, mldata['int0'].T))
        elif name=='nursery':
            raise Exception('Not currently available')
        elif name=='page-blocks':
            data = np.hstack((mldata['target'].T, mldata['data'],
                              mldata['int2'].T))
            target = data[:,-1]
            data = data[:,:-1]
        elif name=='satimage':
            raise Exception('Not currently available')
        elif name=='segment':
            target = mldata['class'].reshape(-1,1)
            data = np.hstack((mldata['int2'].T, mldata['data'],
                              mldata['target'].T, mldata['double3'].T))
        elif name=='vowel':
            target = mldata['Class'].T
            data = np.hstack((
                        self.nominal_to_float(mldata['target'].reshape(-1,1)),
                        mldata['double0'].T,
                        self.nominal_to_float(mldata['data'].reshape(-1,1)),
                        self.nominal_to_float(mldata['Sex'].T.reshape(-1,1))))
        elif name=='zoo':
            target = mldata['type'].reshape(-1,1)
            feature_names = ['aquatic', 'domestic', 'eggs', 'backbone',
                             'feathers', 'data', 'milk', 'tail',
                             'airborne', 'toothed', 'catsize', 'venomous',
                             'fins', 'predator', 'breathes']
            data = np.hstack([
                    self.nominal_to_float(mldata[f_name].reshape(-1,1))
                        for f_name in feature_names])
            data = np.hstack((data, mldata['int0'].T))
        elif name=='abalone':
            target = mldata.target
            data = np.hstack((mldata['data'], mldata['int1'].T))
        elif name=='balance-scale':
            target = mldata.data
            data = mldata.target.T
        elif name=='credit-approval':
            target = mldata['class'].T
            data = self.mldata_to_numeric_matrix(mldata, 690,
                                                 exclude=['class'])
            data, target = self.remove_rows_with_missing_values(data, target)
        else:
            try:
                data = mldata.data
                target = mldata.target
            except KeyError:
                print "KeyError: {}".format(name)
                raise
            except AttributeError:
                print "AttributeError: {}".format(name)
                raise

        return Dataset(name, data, target)

    def mldata_to_numeric_matrix(self, mldata, n_samples, exclude=[]):
        """converts an mldata object into a matrix

        for each value in the mldata dictionary it is reshaped to contain the
        first dimension as a number of samples and the second as number of
        features. If the value contains numerical data it is not preprocessed.
        If the value contains any other type np.object_ it is transformed to
        numerical and all the missing values marked with '?' or 'nan' are
        substituted by np.nan.
        Args:
            mldata (dictionary with some numpy.array): feature strings.

        Returns:
            (array-like, shape = [n_samples, n_features]): floats.
        """
        first_column = True
        for key, submatrix in mldata.iteritems():
            if key not in exclude and type(submatrix) == np.ndarray:
                new_submatrix = np.copy(submatrix)

                if new_submatrix.shape[0] != n_samples:
                    new_submatrix = new_submatrix.T

                if new_submatrix.dtype.type == np.object_:
                    new_submatrix = self.nominal_to_float(new_submatrix)

                if first_column:
                    matrix = new_submatrix.reshape(n_samples, -1)
                    first_column = False
                else:
                    matrix = np.hstack((matrix,
                                        new_submatrix.reshape(n_samples, -1)))
        return matrix


    def nominal_to_float(self, x, missing_values=['nan', '?']):
        """converts an array of nominal features into floats

        Missing values are marked with the string 'nan' and are converted to
        numpy.nan

        Args:
            x (array-like, shape = [n_samples, 1]): feature strings.

        Returns:
            (array-like, shape = [n_samples, 1]): floats.
        """
        new_x = np.empty_like(x, dtype=float)
        x = np.squeeze(x)
        names = np.unique(x)
        substract = 0
        for i, name in enumerate(names):
            if name in missing_values:
                new_x[x==name] = np.nan
                substract += 1
            else:
                new_x[x==name] = i - substract
        return new_x

    def remove_rows_with_missing_values(self, data, target):
        missing = np.logical_or(np.isnan(data),
                                data == -2147483648).any(axis=1)
        data = data[~missing]
        target = target[~missing]
        return data, target


    def sumarize_datasets(self, name=None):
        if name is not None:
            dataset = self.datasets[name]
            dataset.print_summary()
        else:
            for name, dataset in self.datasets.iteritems():
                dataset.print_summary()

def test_datasets(dataset_names):
    from sklearn.svm import SVC
    from sklearn.cross_validation import StratifiedKFold
    data = Data(dataset_names=dataset_names)

    def separate_sets(x, y, test_fold_id, test_folds):
        x_test = x[test_folds == test_fold_id, :]
        y_test = y[test_folds == test_fold_id]

        x_train = x[test_folds != test_fold_id, :]
        y_train = y[test_folds != test_fold_id]
        return [x_train, y_train, x_test, y_test]

    n_folds = 2
    accuracies = {}
    for name, dataset in data.datasets.iteritems():
        dataset.print_summary()
        skf = StratifiedKFold(dataset.target, n_folds=n_folds, shuffle=True)
        test_folds = skf.test_folds
        accuracies[name] = np.zeros(n_folds)
        for test_fold in np.arange(n_folds):
            x_train, y_train, x_test, y_test = separate_sets(
                    dataset.data, dataset.target, test_fold, test_folds)

            svc = SVC(C=1.0, kernel='rbf', degree=1, tol=0.01)
            svc.fit(x_train, y_train)
            prediction = svc.predict(x_test)
            accuracies[name][test_fold] = 100*np.mean((prediction == y_test))
            print("Acc = {0:.2f}%".format(accuracies[name][test_fold]))
    return accuracies

def test():
    dataset_names = ['abalone', 'balance-scale', 'credit-approval']

    not_available_yet = ['derm', 'ecoli',
                     'german', 'heart', 'hepatitis', 'horse', 'iono',
                     'lung-cancer', 'movement', 'mushroom' 'pima', 'satellite',
                     'segmentation', 'spambase', 'wdbc', 'wpbc', 'yeast']

    valid_dataset_names = [name for name in dataset_names if name not in not_available_yet]

    accuracies = test_datasets(valid_dataset_names)
    for name in valid_dataset_names:
        print("{}. {} Acc = {:.2f}% +- {:.2f}".format(
                np.where(np.array(dataset_names) == name)[0]+1,
                name, accuracies[name].mean(), accuracies[name].std()))

if __name__=='__main__':
    test()
