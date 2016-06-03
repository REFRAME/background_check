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
        print('\n')


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
                    #'spam':'uci-20070111 spambase',
                    'tic-tac':'uci-20070111 tic-tac-toe',
                    'MNIST':'MNIST (original)'}

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
        if name == 'spambase':
            data = np.load('./datasets/uci_spambase.npy')
            target = data[:,-1]
            data = data[:,0:-1]
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
