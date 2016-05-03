from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from math import ceil
from math import sqrt
import numpy as np
plt.ion()

class Dataset(object):
    def __init__(self, name, data, target):
        self.name = name
        self._data = data
        self._target, self._classes, self._names = self.standardize_targets(target)

    def standardize_targets(self, target):
        target = np.squeeze(target)
        names = np.unique(target)
        new_target = np.empty_like(target, dtype=int)
        for i, name in enumerate(names):
            new_target[target==name] = i
        classes = range(len(names))
        return new_target, classes, names

    @property
    def target(self):
        return self._target

    @property
    def data(self):
        return self._data

    @property
    def names(self):
        return self._names

    @property
    def classes(self):
        return self._classes


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
                    'waveform-5000':'datasets-UCI waveform-5000'}

    def __init__(self, data_home='./datasets/'):
        self.data_home = data_home
        self.datasets = {}

        for key in MLData.mldata_names.keys():
            self.datasets[key] = self.get_dataset(key)

    def get_dataset(self, name):
        data_home='./datasets/'
        mldata = fetch_mldata(MLData.mldata_names[name], data_home=data_home)

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

    def sumarize_datasets(self):
        for name, dataset in self.datasets.iteritems():
            print('Name = {}'.format(name))
            print('Data shape = {}'.format(dataset.data.shape))
            print('Target shape = {}'.format(dataset.target.shape))
            print('Target classes = {}'.format(dataset.classes))
            print('Target labels = {}'.format(dataset.names))
            print('\n')

