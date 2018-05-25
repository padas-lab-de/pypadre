import copy
import json
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


class VisualizeResults:
    """
    This class visualizes the results of experiments or between different runs of the same experiment
    """

    def __init__(self, path_list=None):
        """
        This class initializes a single visualization instance between two runs
        :param path_list: The os path of the various files that are to be visualized as a list of strings
        """
        self._mean_squared_error = []
        self._data = []
        self._path_list = copy.deepcopy(path_list)
        self._read_paths = []

    def read_directory(self, dir_path=None):
        """
        This function reads all predefined results.json files across multiple runs in an experiment
        :param dir_path: The directory path of the experiment
        :return:
        """
        if not os.path.exists(dir_path):
            return None

        dir_list = self.get_immediate_subdirectories(dir_path)

        for sub_directory in dir_list:
            if os.path.exists(os.path.join(sub_directory, 'results.json')):
                self._path_list.append(os.path.join(sub_directory, 'results.json'))

    def get_immediate_subdirectories(self, dir_path):
        return [os.path.join(dir_path, name) for name in os.listdir(dir_path)
                if os.path.isdir(os.path.join(dir_path, name))]

    def read_data(self):
        """
        This file loads the JSON serialized data into memory
        :return: None
        """
        for path in self._path_list:
            if os.path.exists(path):
                self._read_paths.append(path.split(sep='/')[-2][0:8])
                with open(path, "r") as read_file:
                    data_1 = json.load(read_file)
                    self._data.append(data_1)

            else:
                print('File ', path, ' does not exist, check the input paths')

    def get_confusion_matrix(self):
        """
        This function finds the confusion matrix from the predicted and truth values
        :return: None
        """
        total_error_matrix = np.empty(shape=[len(set(self._data[0]['truth'])), len(self._data)])

        for json_data, row_idx in zip(self._data, range(0, len(self._data))):
            y_true = copy.deepcopy(json_data.get('truth', None))
            y_pred = copy.deepcopy(json_data.get('predicted', None))
            confusion_matrix = metrics.confusion_matrix(y_true=y_true,
                                                        y_pred=y_pred)
            print(confusion_matrix)
            # total test vectors for each class
            total = np.asarray(np.sum(confusion_matrix, axis=0))
            err_val = []
            for idx in range(len(np.asarray(confusion_matrix))):
                err_val.append(np.asarray(confusion_matrix)[idx][idx]/total[idx])
                total_error_matrix[idx][row_idx] = confusion_matrix[idx][idx]/total[idx]

        # Create a bar chart for each label in it.
        idx = 0
        width = 0.25

        # Display a plot with precision grouped by classifier
        ind = np.arange(len(total_error_matrix[0]))
        print(ind)
        rects = []
        label_names = ('A', 'B', 'C')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for row in total_error_matrix:
            print(row)
            rects.append(ax.bar(ind + width * idx, row, width))
            idx = idx + 1

        ax.set_ylabel('Precision')
        ax.set_title('Precision Scores grouped by Label')
        ax.legend(rects, label_names)
        plt.show()

        # Display a bar chart with precision grouped by labels
        fig_labels = plt.figure()
        ax_labels = fig_labels.add_subplot(111)
        rects_transposed = []
        idx = 0
        total_error_matrix_transposed = np.transpose(total_error_matrix)
        width = 0.075
        ind = np.arange(len(total_error_matrix_transposed[0]))
        for row in total_error_matrix_transposed:
            rects_transposed.append(ax_labels.bar(ind + width * idx, row, width))
            idx = idx + 1
        ax_labels.set_ylabel('Precision')
        ax_labels.set_title('Precision Scores grouped by classifier')
        #ax_labels.legend(rects_transposed, label_names)
        plt.show()



    def get_regression_metrics(self):
        """
        This function calculates the different regression metrics
        :return:
        """
        colors = iter(cm.rainbow(np.linspace(0, 1, len(self._data))))
        idx = 0
        for json_data in self._data:
            y_true = copy.deepcopy(json_data.get('truth', None))
            y_pred = copy.deepcopy(json_data.get('predicted', None))

            if y_true is not None and y_pred is not None:
                self._mean_squared_error.append(metrics.mean_squared_error(y_true=y_true,
                                                                            y_pred=y_pred))
                color = next(colors)
                _plt = plt.scatter(y_true, np.asarray(np.absolute(np.subtract(y_true, y_pred))),
                                   color=color, label=self._read_paths[idx])

            idx = idx + 1

        plt.title('Absolute Error')
        plt.ylabel('Absolute error in prediction')
        plt.xlabel('Actual Value')
        plt.legend(loc=1)
        plt.show()


def main():

    # Hard coded paths for comparing between two runs
    path1 = '/home/chris/.pypadre/experiments/Grid_search_experiment_4.ex/db1adfd0-5e70-11e8-b985-080027031794.run/results.json'
    path2 = '/home/chris/.pypadre/experiments/Grid_search_experiment_4.ex/db1adfc4-5e70-11e8-b985-080027031794.run/results.json'

    dir_path = '/home/chris/.pypadre/experiments/Grid_search_experiment_4.ex'
    path_list = []

    # If path_list is not empty all files within the path_list is taken for visualization
    visualize_regression = VisualizeResults(path_list)
    visualize_regression.read_directory(dir_path=dir_path)
    visualize_regression.read_data()
    visualize_regression.get_regression_metrics()

    path_classification = \
        "/home/chris/.pypadre/experiments/Grid_search_experiment_3.ex/53bd15c0-5e86-11e8-b985-080027031794.run" \
        "/results.json"
    dir_path = '/home/chris/.pypadre/experiments/Grid_search_experiment_3.ex'
    path_list.append(path_classification)
    visualize_classification = VisualizeResults(path_list)
    visualize_classification.read_directory(dir_path=dir_path)
    visualize_classification.read_data()
    visualize_classification.get_confusion_matrix()


if __name__ == '__main__':
    main()