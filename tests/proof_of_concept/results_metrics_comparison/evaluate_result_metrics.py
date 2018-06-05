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

        # This dictionary contains the run_ids that have each estimator
        # The estimator name is the key and a list of run_ids will be present
        # that have that particular estimator
        self._run_estimators = dict()

        # This dictionary contains the run_ids corresponding to a particular value
        # of a parameter. key format: estimator_name.param_name.value
        self._param_values_run_id = dict()

        # This dictionary contains a set of all split ids for each run_id
        self._run_split_dict = dict()

        # This dictionary contains the unique parameter names for display
        self._unique_param_names = dict()

        # This dictionary contains the parameters of all the runs
        self._curr_listed_estimators = dict()

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
        # Check for hyperparameters.json, results.json, metadata.json and metrics.json in the folder

        if not(os.path.exists(os.path.join(run_dir, 'hyperparameters.json'))
                or(os.path.exists(os.path.join(run_dir, 'metadata.json')))):
            return  False

        # If there are no split directories return false
        sub_dir = self.get_immediate_subdirectories(run_dir)
        if len(sub_dir) == 0:
            return False

        # If the split directories do not contain metrics.json or results.json return false
        for curr_sub_dir in sub_dir:
            if not(os.path.exists(os.path.join(curr_sub_dir, 'metrics.json'))
                    or(os.path.exists(os.path.join(curr_sub_dir, 'results.json')))):
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

    def get_params(self, run_dir=None):
        """
        Separates the estimator and identifies the params for that estimator
        :param run_dir: A string containing the run folder name
        :return: dictionary containing estimator name and list of params
        """
        if run_dir is None:
            return None

        # Load the hyperparameters.json file from the run directory
        with open(os.path.join(run_dir, "hyperparameter.json"), 'r') as f:
            estimator_parameters = json.loads(f.read())

        hyperparameters = dict()

        # Assumption that in a pipeline one estimator will occur only once.
        for curr_estimator in estimator_parameters:
            parameters = estimator_parameters.get(curr_estimator).get('hyper_parameters').get('model_parameters')
            param_value_dict = dict()
            for curr_param in parameters:
                param_value_dict[curr_param] = parameters.get(curr_param).get('value')

            estimator_name = estimator_parameters.get(curr_estimator).get('algorithm').get('value')
            hyperparameters[estimator_name] = copy.deepcopy(param_value_dict)

        return hyperparameters

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

            # This dictionary contains the default values for each estimator,
            # so that only those parameters that are different across estimators
            # need to be displayed
            estimator_default_values = dict()
            # Aggregate all the hyper parameters in all the run files
            for run_dir in run_dir_list:
                params = self.get_params(run_dir)
                run_id = run_dir[:-4].split(sep='/')[-1]
                self._curr_listed_estimators[run_id] = params

            for run_id in self._curr_listed_estimators:
                estimator_group = self._curr_listed_estimators.get(run_id)
                for estimator in estimator_group:
                    # if the estimator hasn't been seen before, add it to the list
                    if self._unique_estimators.get(estimator, None) is None:
                        # Add the run_id for for the estimator to the dictionary
                        self._run_estimators[estimator] = [run_id]
                        estimator_default_values[estimator] = estimator_group.get(estimator)
                        curr_estimator = estimator_group.get(estimator)
                        # Set the value of each parameter as default
                        param_list = dict()
                        for param in curr_estimator:
                            param_value = frozenset({curr_estimator.get(param)})
                            param_list[param] = param_value
                        self._unique_estimators[estimator] = copy.deepcopy(param_list)

                    else:
                        # Check whether all the values or same or different.
                        # If any param is different add it to the list of differing params
                        # Append the run_id to to list
                        self._run_estimators[estimator].append(run_id)
                        params = estimator_group.get(estimator)
                        default_params = estimator_default_values.get(estimator)
                        for param in params:
                            if default_params.get(param) != params.get(param):
                                param_set = self._unique_estimators.get(estimator).get(param)
                                param_set = param_set.union({params.get(param)})
                                self._unique_estimators[estimator][param] = param_set
                                key = '.'.join([estimator, param, str(params.get(param))])
                                if self._param_values_run_id.get(key, None) is None:
                                    self._param_values_run_id[key] = frozenset({run_id})
                                else:
                                    run_ids = self._param_values_run_id.get(key)
                                    run_ids = run_ids.union({run_id})
                                    self._param_values_run_id[key] = run_ids

        for estimator in self._unique_estimators:
            for param in self._unique_estimators.get(estimator):
                # If len > 1, multiple parameters present, so add it to the list
                if len(self._unique_estimators.get(estimator).get(param))>1:
                    params = self._unique_param_names.get(estimator, None)
                    if params is None:
                        self._unique_param_names[estimator] = frozenset({param})
                    else:
                        self._unique_param_names[estimator] = params.union({param})
                    # The run-id of the runs containing the default params will not be present in
                    # the self._unique_param_values_run_id, so that has to be added to the frozen set
                    # Get the default value of the parameter
                    val = estimator_default_values.get(estimator).get(param)
                    # Check whether if a run_id has the default value and
                    # if it has add it to self._param_values_run_id
                    for run in self._run_estimators.get(estimator):
                        run_params = self._curr_listed_estimators.get(run)
                        if run_params.get(estimator).get(param) == val:
                            key = '.'.join([estimator, param, str(val)])
                            if self._param_values_run_id.get(key, None) is None:
                                self._param_values_run_id[key] = frozenset({run})
                            else:
                                self._param_values_run_id[key] = self._param_values_run_id.get(key).union({run})

        print('Analysis Completed')

    def read_split_metrics(self):
        """
        Reads the metrics.json file from each of the runs
        Creates a dictionary with the key as the run_id,
        and values as split_ids within a run
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
                key = sub_directory[:-6].split(sep='/')[-1]
                self._metrics[key] = data
                run_id = sub_directory[:-6].split(sep='/')[-2][:-4]
                splits = self._run_split_dict.get(run_id, None)
                if splits is None:
                    self._run_split_dict[run_id] = frozenset({key})
                else:
                    self._run_split_dict[run_id] = splits.union({key})


    def display_results(self):
        """
        Displays the collected data as a Pandas data frame
        :return: None
        """
        display_columns = ['run', 'split', 'dataset']
        regression_metrics = ['mean_error', 'mean_absolute_error', 'standard_deviation',
                              'max_absolute_error', 'min_absolute_error']
        classification_metrics = ['accuracy']
        keys = list(self._unique_estimators.keys())
        for key in keys:
            params = self._unique_param_names.get(key)
            if params is not None:
                for param in params:
                    display_columns.append('.'.join([key, param]))

        if self._metrics.get(list(self._metrics.keys())[0]).get('type') == 'regression':
            display_columns = display_columns + regression_metrics
        else:
            display_columns = display_columns + classification_metrics

        data_report = []
        # For all runs
        for run in self._run_split_dict:

            # For all splits in a run
            for split in self._run_split_dict.get(run):
                # run_id split_id dataset
                data_row = dict()
                data_row['run'] = run
                data_row['split'] = split

                metrics = self._metrics.get(split)
                data_row['dataset'] = metrics.get('dataset', 'NA')

                # Unique estimators are present in keys
                # Check whether the current run is having the estimator
                for key in keys:
                    if run in self._run_estimators.get(key, None) is not None:
                        params = self._unique_param_names.get(key, None)
                        if params is not None:
                            for param in params:
                                val = self._curr_listed_estimators.get(run)
                                data_row['.'.join([key, param])] = val.get(key).get(param, '-')

                if metrics.get('type', None) == 'regression':
                    for metric in regression_metrics:
                        data_row[metric] = metrics.get(metric, '-')

                else:
                    for metric in classification_metrics:
                        data_row[metric] = metrics.get(metric, '-')

                curr_tuple = tuple()
                for col in display_columns:
                    curr_tuple = curr_tuple + tuple([str(data_row.get(col, '-'))])

                data_report.append(copy.deepcopy(curr_tuple))


        df = pd.DataFrame(data=data_report)
        df.columns = display_columns
        print(df)


def main():

    # Load the results folder
    dir_path = filedialog.askdirectory(initialdir="~/.pypadre/experiments", title="Select Experiment Directory")
    # It could either be experiments in a directory or multiple experiments
    dir_list = list()
    if len(dir_path) == 0:
        return
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
