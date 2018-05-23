import copy
import json
import os

import matplotlib.pyplot
import numpy as np
from sklearn import metrics


class VisualizeResults():
    """
    This class visualizes the results of experiments or between different runs of the same experiment
    """

    def __init__(self, path_to_run1, path_to_run2):
        """
        This class initializes a single visualization instance between two runs
        :param path_to_run1: The os path to the results file of the first run
        :param path_to_run2: The os path to the results file of the second run
        """
        self._path1 = path_to_run1
        self._path2 = path_to_run2

    def read_data(self):
        """
        This file loads the JSON serialized data into memory
        :return: None
        """
        if os.path.exists(self._path1) and \
           os.path.exists(self._path1):

            with open(self._path1, "r") as read_file:
                self._data_1 = json.load(read_file)

            with open(self._path2, "r") as read_file:
                self._data_2 = json.load(read_file)

        else:
            print('File does not exist, check the input paths')

    def get_confusion_matrix(self):
        """
        This function finds the confusion matrix from the predicted and truth values
        :return: None
        """
        #self._confusion_matrix = metrics.confusion_matrix(y_true=a,
        #                                                  y_pred=b)

    def get_regression_metrics(self):
        """
        This function calculates the different regression metrics
        :return:
        """
        y_true = copy.deepcopy(self._data_1.get('truth', None))
        y_pred = copy.deepcopy(self._data_1.get('predicted', None))

        if y_true is not None and y_pred is not None:
            self._mean_squared_error_data_1 = metrics.mean_squared_error(y_true=y_true,
                                                                        y_pred=y_pred)
            matplotlib.pyplot.scatter(y_true, np.asarray(np.subtract(y_true, y_pred)),
                                      c="b")
            print(self._mean_squared_error_data_1)

        y_true = copy.deepcopy(self._data_2.get('truth', None))
        y_pred = copy.deepcopy(self._data_2.get('predicted', None))
        if y_true is not None and y_pred is not None:
            self._mean_squared_error_data_2 = metrics.mean_squared_error(y_true=y_true,
                                                                        y_pred=y_pred)
            matplotlib.pyplot.scatter(y_true, np.asarray(np.subtract(y_true, y_pred)),
                                      c="r")
            print(self._mean_squared_error_data_2)
        matplotlib.pyplot.show()





def main():
    path1 = '/home/chris/.pypadre/experiments/Grid_search_experiment_4.ex/db1adfd0-5e70-11e8-b985-080027031794.run/results.json'
    path2 = '/home/chris/.pypadre/experiments/Grid_search_experiment_4.ex/db1adfc4-5e70-11e8-b985-080027031794.run/results.json'

    visualize = VisualizeResults(path1, path2)
    visualize.read_data()
    visualize.get_regression_metrics()



if __name__ == '__main__':
    main()