import copy
import json
import os
from tkinter import filedialog

import pandas as pd


class CompareMetrics:

    def __init__(self, dir_path=None, file_path=None):
        """
        The constructor that initializes the object with all the required paths
        :param dir_path: The path of the experiment to be compared
        :param file_path: The path of the JSON file to be compared
        """
        if dir_path is not None:
            self._dir_path = dir_path

        if file_path is not None:
            self._file_path = copy.deepcopy(file_path)

        self._run_dir = []
        self._param_values = dict()
        self._metrics = dict()
        self._unique_estimators = dict()

    def get_immediate_subdirectories(self, dir_path):
        """
        Gets the immediate subdirectories present in the dir_path
        :param dir_path: The current experiment directory to be checked
        :return: A list of run directories present in the experiment directory
        """
        return [os.path.join(dir_path, name) for name in os.listdir(dir_path)
                if os.path.isdir(os.path.join(dir_path, name))]

    def validate_run_dir(self, run_dir):
        """
        Validates whether the given directory is a valid run directory of an experiment
        :param run_dir: The path of the run directory
        :return: True: If it is valid, False otherwise
        """
        # run_dir format: '/home/user_name/.pypadre/experiments/Grid_search_experiment_3.ex/
        # SVC[C(0.5)degr(2)].run'
        import re
        if not run_dir[-4:] == '.run':
            return False
        expr = re.compile(r'^[\w]+\[(\w{1,4}\(\w*\.?\w+\))+\]', re.IGNORECASE)
        param_list = run_dir[:-4].split('/')[-1].split(sep=';')
        for param in param_list:
            if expr.fullmatch(param) is None:
                print('Listed directory is not a valid run directory:' + param)
                return False

        return True

    def read_run_directories(self):
        """
        unique_estimators = dict()
        This function reads all the run directories and stores the names,
        for obtaining the parameters that changed
        :return: None
        """

        if self._dir_path is None:
            return

        for curr_experiment in self._dir_path:
            if not os.path.exists(curr_experiment):
                continue

            run_dir_list = self.get_immediate_subdirectories(curr_experiment)

            # Check whether each path is a valid run format
            for run_dir in run_dir_list:
                if not self.validate_run_dir(run_dir=run_dir):
                    continue

                self._run_dir.append(copy.deepcopy(run_dir))

    def get_estimators_parameter_values(self):
        """
        Gets all the unique estimators and their parameters from the saved run directories
        :return: None
        """
        for curr_run in self._run_dir:
            # Get the param values for every run.
            # Every split within a run has the same params,
            # differing only by the testing and training rows
            key = '$'.join(curr_run.split(sep='/')[-2:])
            value_dict = self.get_param_values(curr_run.split(sep='/')[-1][:-4])
            self._param_values[key] = value_dict

    def get_param_values(self, run_name=None):
        if run_name is None:
            return
        estimator_dict = dict()
        # Strip the run name into multiple estimators separated by ;
        estimator_param_list = run_name.split(sep=';')
        for estimator_param in estimator_param_list:
            # Identify the estimator and get the parameters corresponding to that estimator
            estimator_name = estimator_param[0:estimator_param.find('[')]
            param_values_list = estimator_param[estimator_param.find('[') + 1:-1].split(sep=')')
            param_dict = dict()
            # Add each parameter and its value to the dictionary
            for param_value in param_values_list:
                if param_value == '':
                    continue
                param_name = param_value[0:param_value.find('(')]
                param_value = param_value[param_value.find('(') + 1:]
                param_dict[param_name] = param_value

            estimator_dict[estimator_name] = copy.deepcopy(param_dict)

        return estimator_dict

    def get_params(self, estimator_params=None):
        """
        Separates the estimator and identifies the params for that estimator
        :param estimator_params: A string containing the run folder name
        describing all estimators and its parameters
        format: estimator_name1[param1(value)param2(value)];estimator_name2[param3(value)param4(value)]...
        :return: dictionary containing estimator name and list of params
        """
        if estimator_params is None:
            return None

        param_dict = dict()
        # The estimator_params is of the form 'SVR[C(0.5)degr(1)];pca[n_co(2)]'
        # SVR : estimator name
        #       C : parameter name with value 0.5
        #       degr: parameter_name with value 2
        # pca: estimator name
        #      n_co: parameter name with value 2
        estimators = estimator_params.split(sep=';')
        for estimator in estimators:
            estimator_name = estimator[:estimator.find('[')]
            param_strings = estimator[estimator.find('[') + 1:-1].split(sep=')')
            param_list = []
            for param in param_strings:
                if len(param) == 0:
                    continue
                param_list.append(param[:param.find('(')])

            param_dict[estimator_name] = param_list

        return param_dict

    def get_unique_estimators_parameter_names(self):
        """
        Gets all the unique estimators and their parameter names from the saved run directories
        :return: None
        """
        # Get a single run directory from the experiment to identify the parameters that vary
        # in that experiment
        for curr_experiment in self._dir_path:
            if not os.path.exists(curr_experiment):
                continue

            run_dir_list = self.get_immediate_subdirectories(curr_experiment)

            curr_estimators = list
            # Check whether each path contains a .run
            # And then get the estimators and corresponding parameters of that experiment
            for run_dir in run_dir_list:
                # run_dir format: '/home/user_name/.pypadre/experiments/Grid_search_experiment_3.ex/
                # SVC[C(0.5)degr(2)].run'
                curr_estimators = self.get_params(run_dir[:-4].split('/')[-1])
                break

            for estimator in curr_estimators:
                # If the curr param is not present in the unique estimators then,
                # add it to the unique_estimators dictionary
                if self._unique_estimators.get(estimator, None) is None:
                    self._unique_estimators[estimator] = copy.deepcopy(curr_estimators.get(estimator))
                # Check whether all the parameters for that estimator are present
                # If a new parameter is found, add it to the current known list
                else:
                    known_params = self._unique_estimators.get(estimator)
                    curr_params = curr_estimators.get(estimator)
                    new_params = list(set(curr_params) - set(known_params))
                    if len(new_params) > 0:
                        known_params.append(new_params)
                        self._unique_estimators[estimator] = copy.deepcopy(known_params)

    def read_split_metrics(self):
        """
        Reads the metrics.json file from each of the runs
        :return: None
        """

        for curr_run_dir in self._run_dir:
            sub_directory_list = self.get_immediate_subdirectories(curr_run_dir)
            for sub_directory in sub_directory_list:
                if sub_directory[-6:] != '.split':
                    continue
                # Check if a metrics file and a results file is located within the split directory
                if not os.path.exists(os.path.join(sub_directory, 'metrics.json')):
                    continue

                # read the json file into memory
                with open(os.path.join(sub_directory, 'metrics.json'), "r") as read_file:
                    data = json.load(read_file)
                key = '$'.join(sub_directory[:-6].split(sep='/')[-3:])
                print(key)
                self._metrics[key] = data

    def display_results(self):
        """
        Displays the collected data as a Pandas data frame
        :return: None
        """
        # The dictionary needs only the accuracy for now
        display_dict = dict()
        print(self._unique_estimators)
        for item in self._metrics:
            data_dict = dict()
            data = self._metrics.get(item)
            estimators_params = self._param_values.get('$'.join(item.split('$')[0:-1]))
            #data_dict['params'] = estimators_params
            for estimator in estimators_params:
                params_list = estimators_params.get(estimator)
                for param in params_list:
                   data_dict['.'.join([estimator,param])] = params_list.get(param)
                print(estimator)

            data_dict['accuracy'] = data.get('accuracy')
            display_dict[item] = copy.deepcopy(data_dict)
        data_frame = pd.DataFrame.from_dict(display_dict, orient='index')
        print(data_frame)
        # train = pd.DataFrame.from_dict(self._metrics, orient='index')


def main():

    # Load the results folder
    dir_path = filedialog.askdirectory(initialdir="~/.pypadre/experiments", title="Select Experiment Directory")
    # It could either be experiments in a directory or multiple experiments
    dir_list = list()
    dir_list.append(dir_path)

    metrics = CompareMetrics(dir_path=dir_list)
    metrics.read_run_directories()
    # From the run directory names, obtain the estimators and the parameters
    metrics.get_unique_estimators_parameter_names()
    metrics.get_estimators_parameter_values()
    # Read the JSON file objects from the .split folders
    metrics.read_split_metrics()
    # Display the results using Pandas data frame
    metrics.display_results()


if __name__ == '__main__':
    main()
