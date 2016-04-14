"""
This class can be used to save the intermediate results of any experiment
run in python. It will create a folder for the specific experiment and
save all the information and images in a structured and sorted manner.
"""
__docformat__ = 'restructedtext en'
import os
import sys
import datetime

import csv

try:
    import PIL.Image as Image
except ImportError:
    import Image


class Notebook(object):
    def __init__(self, name, path):
        self.name = name
        self.filename = "{}.csv".format(name)
        self.path = path
        self.entry_number = 0

    def add_entry(self, row, general_entry_number=0):
        self.entry_number += 1
        with open(os.path.join(self.path, self.filename), 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|',
                    quoting=csv.QUOTE_NONNUMERIC)
            now = datetime.datetime.now()
            writer.writerow([general_entry_number, self.entry_number,
                             now.date(), now.time()] + row)

class Diary(object):

    __DESCR_FILENAME='description.txt'

    def __init__(self, name, path='diary', overwrite=False, image_format='png',
                 fig_format='svg'):
        self.creation_date = datetime.datetime.now()
        self.name = name
        self.path = os.path.join(path,name)
        self.overwrite = overwrite

        self.image_format = image_format
        self.fig_format = fig_format
        self.entry_number = 0

        self.all_paths = self._create_all_paths()
        self._save_description()

        self.notebooks = {}

    def add_notebook(self, name):
        self.notebooks[name] = Notebook(name, self.path)

    def _create_all_paths(self):
        path = self.path
        i = 0
        while self.overwrite == False and os.path.exists(self.path):
            self.path = "{}_{}".format(path,i)
            i +=1

        self.path_images = os.path.join(self.path, 'images')
        self.path_figures = os.path.join(self.path, 'figures')
        all_paths = [self.path, self.path_images, self.path_figures]
        for path in all_paths:
            if not os.path.exists(path):
                os.makedirs(path)
        return all_paths

    def _save_description(self):
        with open(os.path.join(self.path, self.__DESCR_FILENAME), 'w') as f:
            print("Writting :\n{}".format(self))
            f.write(self.__str__())

    def add_entry(self, notebook_name, row):
        self.entry_number += 1
        self.notebooks[notebook_name].add_entry(row, self.entry_number)

    def save_image(self, image, filename='', extension=None):
        if extension == None:
            extension = self.image_format
        image.save(os.path.join(self.path_images,
                                "{}_{}.{}".format(filename, self.entry_number,
                                                  extension)))

    # TODO add support to matplotlib.pyplot.figure or add an additional
    # function
    def save_figure(self, plt, filename='', extension=None):
        if extension == None:
            extension = self.fig_format
        plt.savefig(os.path.join(self.path_figures,
                                "{}_{}.{}".format(filename, self.entry_number,
                                                  extension)))

    def __str__(self):
        return ("Date: {}\nName : {}\nPath : {}\n"
                "Overwrite : {}\nImage_format : {}\n"
                "").format(self.creation_date, self.name, self.path,
                        self.overwrite, self.image_format)

if __name__ == "__main__":
    diary = Diary(name='world', path='hello', overwrite=False)

    diary.add_notebook('validation')
    diary.add_notebook('test')

    diary.add_entry('validation', ['accuracy', 0.3])
    diary.add_entry('validation', ['accuracy', 0.5])
    diary.add_entry('validation', ['accuracy', 0.9])
    diary.add_entry('test', ['First test went wrong', 0.345, 'label_1'])

    image = Image.new(mode="1", size=(16,16), color=0)
    diary.save_image(image, filename='test_results')
