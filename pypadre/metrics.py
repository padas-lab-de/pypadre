"""
Classes for managing the metrics related to experiments.
The metrics can be visualized as pandas dataframes or the metrics
could be recalculated.
"""

import copy
import json
import os

import numpy as np
import pandas as pd

from pypadre.eventhandler import trigger_event, assert_condition

REGRESSION = 'regression'
CLASSIFICATION = 'classification'

class ReevaluationMetrics:
    """
    This class reevaluates the metrics from the results.json file present in the split folder.
    The user should be able to specify the required metrics and those additional metrics will be computed, if
    the functions for those metrics are available.
    """
    _dir_path = []

    def __init__(self, dir_path=None, file_path=None):
        """
        Initilization function with the paths of the run directories whose metrics are to be reevaluated.

        :param dir_path: The directories of the run paths whose metrics have to be reevaluated.
        :param file_path: The exact file path whose metrics have to be evaluated.

        TODO: Implement the functionality where the exact files could be given and results are recomputed only for those
        """
        if dir_path is not None:
            self._dir_path = dir_path

        if file_path is not None:
            self._file_path = file_path

        self._split_dir = []

    def get_immediate_subdirectories(self, dir_path):
        """
        Gets the immediate subdirectories present in the dir_path.

        :param dir_path: The current experiment directory to be checked.

        :return: A list of run directories present in the experiment directory.
        """
        return [os.path.join(dir_path, name) for name in os.listdir(dir_path)
                if os.path.isdir(os.path.join(dir_path, name))]

    def get_split_directories(self, dir_path=None):
        """
        Finds all the split directories within the path.

        :param dir_path: An optional directory path list to be added for recomputing metrics.

        :return: None
        """

        if dir_path is not None:
            self._dir_path = copy.deepcopy(self._dir_path.append(dir_path))

        for experiment_path in self._dir_path:
            run_dir_list = self.get_immediate_subdirectories(experiment_path)

            for run_path in run_dir_list:
                split_dir_list = self.get_immediate_subdirectories(run_path)

                for split_dir in split_dir_list:
                    if os.path.exists(os.path.join(split_dir, 'results.json')):
                        self._split_dir.append(split_dir)

    def recompute_metrics(self):
        """
        The function recomputes the metrics based on the inputs from the user.

        :return: None
        """

        for split_path in self._split_dir:

            # Load the hyperparameters.json file from the run directory
            with open(os.path.join(split_path, "results.json"), 'r') as f:
                results = json.loads(f.read())

            prediction_type = results.get('type', None)
            dataset = results.get('dataset', '-')
            metrics = None
            assert_condition(condition=prediction_type is None, source=self,
                             message='No prediction type present in ' + split_path)

            if prediction_type == REGRESSION:
                metrics = self.compute_regression_metrics(results)

            elif prediction_type == CLASSIFICATION:
                metrics = self.compute_classification_metrics(results)

            assert_condition(condition=metrics is not None, source=self,
                             message='Error reading the results.json file in ' + split_path)

            if metrics is not None:
                metrics['type'] = prediction_type
                metrics['dataset'] = dataset
                with open(os.path.join(split_path, "metrics.json"), 'w') as f:
                    f.write(json.dumps(metrics))

    def compute_regression_metrics(self, results=None):
        """
        Function to compute the regression metrics of a split from the result.

        :param results: The predicted value and the ground truth values in a dictionary.

        :return: A dictionary containing the computed regression metrics.
        """

        assert_condition(condition=results is not None, source=self, message='Results are None')

        y_true = results.get('truth', None)
        y_predicted = results.get('predicted', None)

        assert_condition(condition=not(y_true is None or y_predicted is None), source=self,
                         message='Truth or Predicted values missing in results.json')

        metrics_dict = dict()
        y_true = np.array(y_true)
        y_predicted = np.array(y_predicted)
        error = y_true - y_predicted
        metrics_dict['mean_error'] = np.mean(error)
        metrics_dict['mean_absolute_error'] = np.mean(abs(error))
        metrics_dict['standard_deviation'] = np.std(error)
        metrics_dict['max_absolute_error'] = np.max(abs(error))
        metrics_dict['min_absolute_error'] = np.min(abs(error))
        return copy.deepcopy(metrics_dict)

    def compute_classification_metrics(self, results=None):
        """
        Function to compute the classification metrics of a split from the results.

        :param results: A dictionary containing the truth value and the predicted value.

        :return: A dictionary containing the computed classification metrics
        """

        assert_condition(condition=results is not None, source=self, message='Results are empty')

        y_true = results.get('truth', None)
        y_predicted = results.get('predicted', None)
        option = results.get('average', 'macro')

        assert_condition(condition=not(y_true is None or y_predicted is None), source=self,
                         message='Predicted/Actual results are not available')

        confusion_matrix = self.compute_confusion_matrix(predicted=y_predicted,
                                                         truth=y_true)

        assert_condition(condition=confusion_matrix is not None, source=self,
                         message='Could not compute confusion matrix')

        classification_metrics = dict()
        classification_metrics['confusion_matrix'] = confusion_matrix
        length = len(confusion_matrix)
        precision = np.zeros(shape=length)
        recall = np.zeros(shape=length)
        f1_measure = np.zeros(shape=length)
        tp = 0
        column_sum = np.sum(confusion_matrix, axis=0)
        row_sum = np.sum(confusion_matrix, axis=1)
        for idx in range(0, length):
            tp = tp + confusion_matrix[idx][idx]
            # Removes the 0/0 error
            precision[idx] = np.divide(confusion_matrix[idx][idx], column_sum[idx] + int(column_sum[idx] == 0))
            recall[idx] = np.divide(confusion_matrix[idx][idx], row_sum[idx] + int(row_sum[idx] == 0))
            if recall[idx] == 0 or precision[idx] == 0:
                f1_measure[idx] = 0
            else:
                f1_measure[idx] = 2 / (1.0 / recall[idx] + 1.0 / precision[idx])

        accuracy = tp / np.sum(confusion_matrix)
        if option == 'macro':
            classification_metrics['recall'] = float(np.mean(recall))
            classification_metrics['precision'] = float(np.mean(precision))
            classification_metrics['accuracy'] = accuracy
            classification_metrics['f1_score'] = float(np.mean(f1_measure))

        elif option == 'micro':
            # Micro average is computed as the total number of true positives to the total number of instances
            classification_metrics['recall'] = float(tp/len(y_true))
            classification_metrics['precision'] = float(tp/len(y_true))
            classification_metrics['accuracy'] = accuracy
            classification_metrics['f1_score'] = float(tp/len(y_true))

        else:
            classification_metrics['recall'] = recall.tolist()
            classification_metrics['precision'] = precision.tolist()
            classification_metrics['accuracy'] = accuracy
            classification_metrics['f1_score'] = f1_measure.tolist()

        return copy.deepcopy(classification_metrics)

    def compute_confusion_matrix(self, predicted=None,
                                 truth=None):
        """
        This function computes the confusion matrix of a classification result.
        This was done as a general purpose implementation of the confusion_matrix.

        :param predicted: The predicted values of the confusion matrix.

        :param truth: The truth values of the confusion matrix.

        :return: The confusion matrix.
        """
        import copy
        if predicted is None or truth is None or \
                len(predicted) != len(truth):
            return None

        # Get the number of labels from the predicted and truth set
        label_count = len(set(predicted).union(set(truth)))
        confusion_matrix = np.zeros(shape=(label_count, label_count), dtype=int)
        # If the labels given do not start from 0 and go up to the label_count - 1,
        # a mapping function has to be created to map the label to the corresponding indices
        if (min(predicted) != 0 and min(truth) != 0) or \
                (max(truth) != label_count - 1 and max(predicted) != label_count - 1):
            labels = list(set(predicted).union(set(truth)))
            for idx in range(0, len(truth)):
                row_idx = int(labels.index(truth[idx]))
                col_idx = int(labels.index(predicted[idx]))
                confusion_matrix[row_idx][col_idx] += 1

        else:

            # Iterate through the array and update the confusion matrix
            for idx in range(0, len(truth)):
                confusion_matrix[int(truth[idx])][int(predicted[idx])] += 1

        return copy.deepcopy(confusion_matrix.tolist())


class CompareMetrics:
    """
    This class is used to compare the results of different experiments. It could be based on a single experiment,
    in which case the runs of only that experiment are considered or multiple experiments where the runs of all the
    experiments input are considered.
    """

    _dir_path = []
    _root_path = ''
    _experiments = None

    def __init__(self, dir_path=None, file_path=None, experiments_list=None, root_path=None):
        """
        The constructor that initializes the object with all the required paths.

        :param dir_path: The path of the experiment to be compared.
        :param file_path: The path of the JSON file to be compared.

        TODO: Implement the functionality to compare metrics based on a list of file paths.
        """
        if dir_path is not None:
            self._dir_path = dir_path

        if file_path is not None:
            self._file_path = copy.deepcopy(file_path)

        if experiments_list is not None:
            self._experiments = experiments_list

        if root_path is not None:
            self._root_path = root_path

        self._run_dir = []
        # This dictionary contains the metrics read from all the metrics.json file
        # The key is the split id
        self._metrics = dict()

        # This dictionary contains all the values for each param for every estimator
        # The key is the estimator name
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

        # This dictionary keeps track of the runs to be displayed
        self._display_run = dict()

        # The mandatory fields while displaying results
        self._mandatory_display_columns = ['run', 'split', 'dataset']

        # The metrics to displayed
        self._metrics_display = []

    def get_immediate_subdirectories(self, dir_path):
        """
        Gets the immediate subdirectories present in the dir_path.

        :param dir_path: The current experiment directory to be checked.

        :return: A list of run directories present in the experiment directory.
        """
        return [os.path.join(dir_path, name) for name in os.listdir(dir_path)
                if os.path.isdir(os.path.join(dir_path, name))]

    def validate_run_dir(self, run_dir):
        """
        Validates whether the given directory is a valid run directory of an experiment.

        :param run_dir: The path of the run directory.

        :return: True: If it is valid, False otherwise.
        """
        # Check for hyperparameters.json, results.json, metadata.json and metrics.json in the folder

        if not (os.path.exists(os.path.join(run_dir, 'hyperparameters.json'))
                or (os.path.exists(os.path.join(run_dir, 'metadata.json')))):
            return False

        # If there are no split directories return false
        sub_dir = self.get_immediate_subdirectories(run_dir)
        if len(sub_dir) == 0:
            return False

        # If the split directories do not contain metrics.json or results.json return false
        for curr_sub_dir in sub_dir:
            if not (os.path.exists(os.path.join(curr_sub_dir, 'metrics.json'))
                    or (os.path.exists(os.path.join(curr_sub_dir, 'results.json')))):
                return False

        return True

    def read_run_directories(self, run_dir=None):
        """
        This function reads all the run directories and stores the names,
        for obtaining the parameters that changed.

        :param: run_dir: The path of the experiments as a list.

        :return: None
        """

        if run_dir is not None:
            self._dir_path = copy.deepcopy(self._dir_path + run_dir)

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

    def get_params(self, run_dir=None):
        """
        Separates the estimator and identifies the params for that estimator.

        :param run_dir: A string containing the run folder name.

        :return: dictionary containing estimator name and list of params.
        """
        assert_condition(condition=run_dir is not None, source=self, message='Run directory value is None')

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

    def get_unique_estimators_parameter_names_from_directories(self):
        """
        Gets all the unique estimators and their parameter names from the saved run directories.

        :return: None
        """

        # This dictionary contains the default values for each estimator,
        # so that only those parameters that are different across estimators
        # need to be displayed
        estimator_default_values = dict()

        if isinstance(self._dir_path, list) and len(self._dir_path) == 0:
            return

        # Get a single run directory from the experiment to identify the parameters that vary
        # in that experiment
        for curr_experiment in self._dir_path:
            if not os.path.exists(curr_experiment):
                continue

            run_dir_list = self.get_immediate_subdirectories(curr_experiment)

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
                        if default_params is None:
                            estimator_default_values[estimator] = estimator_group.get(estimator)
                            default_params = estimator_default_values.get(estimator)
                        for param in params:
                            if default_params.get(param) != params.get(param):
                                param_set = self._unique_estimators.get(estimator).get(param)
                                param_set = param_set.union({params.get(param)})
                                self._unique_estimators[estimator][param] = param_set
                                # Add the run id to the dictionary having estimator.param_name.param_value
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
                if len(self._unique_estimators.get(estimator).get(param)) > 1:
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

    def get_unique_estimators_parameter_names_from_experiments(self):
        """
        Gets all the unique estimators and their parameter names from the experiments list

        :return: None
        """

        # This dictionary contains the default values for each estimator,
        # so that only those parameters that are different across estimators
        # need to be displayed
        estimator_default_values = dict()

        if isinstance(self._experiments, list) and len(self._experiments) == 0:
            return

        # Get a single run directory from the experiment to identify the parameters that vary
        # in that experiment
        for curr_experiment in self._experiments:

            # Aggregate all the hyper parameters in all the run files
            idx = 0
            for run_dir in list(curr_experiment.run_split_dict.keys()):
                params = curr_experiment.hyperparameters_combinations[idx][0]
                if len(params) == 0:
                    continue
                run_id = run_dir[:-4].split(sep='/')[-1]
                self._curr_listed_estimators[run_id] = params
                idx += 1

            if len(self._curr_listed_estimators) == 0:
                continue

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
                            # When running the show metrics function back to back, the unique estimators will be
                            # populated while the default params will be empty. This causes a crash.
                            # So if the default values are not populated, then add them to the dictionary
                            if default_params is None:
                                estimator_default_values[estimator] = estimator_group.get(estimator)
                                default_params = estimator_default_values.get(estimator)

                            if default_params.get(param) != params.get(param):
                                param_set = self._unique_estimators.get(estimator).get(param)
                                param_set = param_set.union({params.get(param)})
                                self._unique_estimators[estimator][param] = param_set
                                # Add the run id to the dictionary having estimator.param_name.param_value
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
                if len(self._unique_estimators.get(estimator).get(param)) > 1:
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

    def get_unique_estimators_parameter_names(self):
        """

        :return:
        """

        if self._dir_path is not None or (isinstance(self._dir_path, list) and len(self._dir_path) == 0):
            self.get_unique_estimators_parameter_names_from_directories()

        if self._experiments is not None or (isinstance(self._experiments, list) and len(self._experiments) == 0):
            self.get_unique_estimators_parameter_names_from_experiments()

    def read_split_metrics_from_directories(self):
        """
        Reads the metrics.json file from each of the runs
        Creates a dictionary with the key as the run_id,
        and values as split_ids within a run.

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

        self._display_run = copy.deepcopy(list(self._run_split_dict.keys()))

    def read_split_metrics_from_experiments(self):
        """
        Reads the metrics.json file from each of the experiment runs
        Creates a dictionary with the key as the run_id,
        and values as split_ids within a run.

        :return: None
        """
        for curr_experiment in self._experiments:
            runs = list(curr_experiment.run_split_dict.keys())
            for curr_run in runs:
                splits = curr_experiment.run_split_dict.get(curr_run, None)
                for split_id in splits:
                    # read the metrics dictionary into memory
                    key = split_id[:-6]
                    run_idx = runs.index(curr_run)
                    split_idx = splits.index(split_id)
                    self._metrics[key] = curr_experiment.metrics[run_idx][split_idx]
                    run_id = curr_run[:-4]
                    split = self._run_split_dict.get(run_id, None)
                    # Add the split after removing the .split characters to the run_split dictionary
                    if split is None:
                        self._run_split_dict[run_id] = frozenset({key})
                    else:
                        self._run_split_dict[run_id] = split.union({key})

        self._display_run = copy.deepcopy(list(self._run_split_dict.keys()))

    def read_split_metrics(self):
        if self._dir_path is not None or (isinstance(self._dir_path, list) and len(self._dir_path) == 0):
            self.read_split_metrics_from_directories()

        if self._experiments is not None or (isinstance(self._experiments, list) and len(self._experiments) == 0):
            self.read_split_metrics_from_experiments()

    def compute_results(self):
        """
        Computes the collected data as a Pandas data frame.

        :return: The pandas dataframe that is to be displayed
        """
        display_columns = copy.deepcopy(self._mandatory_display_columns)
        regression_metrics = ['mean_error', 'mean_absolute_error', 'standard_deviation',
                              'max_absolute_error', 'min_absolute_error']
        classification_metrics = ['accuracy', 'f1_score', 'recall', 'precision']
        keys = list(self._unique_estimators.keys())
        for key in keys:
            params = self._unique_param_names.get(key)
            if params is not None:
                for param in params:
                    display_columns.insert(3, '.'.join([key, param]))

        if len(self._metrics_display) == 0:
            if self._metrics.get(list(self._metrics.keys())[0]).get('type') == REGRESSION:
                display_columns = display_columns + regression_metrics
            else:
                display_columns = display_columns + classification_metrics

        else:
            display_columns = display_columns + self._metrics_display

        data_report = []
        # For all runs that need to be displayed
        assert_condition(condition=self._display_run is not None, source=self,
                         message='No runs to be displayed')

        for run in self._display_run:

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

                if metrics.get('type', None) == REGRESSION:
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
        if len(data_report) > 0 and len(data_report[0]) == len(display_columns):
            df.columns = display_columns
        return df

    def analyze_runs(self, query=None, metrics=None, options=None):
        """
        This function would return a pandas data frame based on the query
        and the metrics required.

        :param query: Initially, this would be a list of directories to be evaluated upon
        :param metrics: The metrics to be displayed
        :param options: Any other option possible, like micro averaged, macro averaged etc

        :return: A pandas data frame containing the results of the query
        
        TODO: The options functionality is not yet implemented
        """
        assert_condition(condition=query is not None, source=self, message='Function called with empty query')

        row_id_set = set()

        for element in query:

            if str(element).lower() == 'all':
                self._display_run = copy.deepcopy(self._run_split_dict)
                return

            assert_condition(condition=self._run_estimators.get(element, None) is not None or
                             self._param_values_run_id.get(element, None) is not None,
                             source=self, message='Element ' + element + ' not present ')

            if self._run_estimators.get(element, None) is not None:
                row_id_set = row_id_set.union(set(self._run_estimators.get(element, None)))

            elif self._param_values_run_id.get(element, None) is not None:
                row_id_set = row_id_set.union(set(self._param_values_run_id.get(element, None)))

            else:
                # Query not present within lists
                pass

        self._display_run = copy.deepcopy(list(row_id_set))

        self._metrics_display = []
        if metrics is not None:
            for metric in metrics:
                self._metrics_display.append(metric)

    def get_unique_estimator_names(self):
        """
        This function returns the unique estimators among the runs.

        :return: A list of the names of unique estimators
        """
        return copy.deepcopy(list(self._unique_estimators.keys()))

    def get_estimator_param_values(self, estimator_name, selected_params=None, return_all_values=False):
        """
        This function returns all the parameter values present among the runs for a particular estimator.

        :param estimator_name: he name of the estimator whose differing parameters are to be fetched.
        :param selected_params: Displays the parameter values of the specified parameters only.
        :param return_all_values: whether a list of the default values are also to be returned.

        :return: List of param values, None if the estimator is not present
        """
        params_list = dict()
        params = self._unique_estimators.get(estimator_name, None)
        assert_condition(condition=params is not None, source=self,
                         message='No parameters obtained for estimator :' + estimator_name)

        # If all the parameters are selected, then return the whole param dictionary
        if return_all_values:
            return copy.deepcopy(params)

        # If only selected params are given, then search for those params and return them
        if selected_params is None:
            selected_params = self._unique_param_names.get(estimator_name, None)

        # selected_params will contain the parameters & values of a single estimator to be shown to the user
        # if a selected estimator does not have any unique parameters, it will be none
        if selected_params is not None:
            for param in selected_params:
                values = params.get(param, None)
                if values is not None:
                    params_list[param] = values

        return copy.deepcopy(params_list)

    def display_results(self):
        """
        Displays the collected data as a Pandas data frame.

        :return: None
        """
        metrics = None
        display_columns = copy.deepcopy(self._mandatory_display_columns)
        regression_metrics = ['mean_error', 'mean_absolute_error', 'standard_deviation',
                              'max_absolute_error', 'min_absolute_error']
        classification_metrics = ['accuracy', 'f1_score', 'recall', 'precision']
        keys = list(self._unique_estimators.keys())
        for key in keys:
            params = self._unique_param_names.get(key)
            if params is not None:
                for param in params:
                    display_columns.insert(3, '.'.join([key, param]))

        # Return if there are no metrics to be displayed
        if len(self._metrics) == 0:
            return

        if len(self._metrics_display) == 0:
            if self._metrics.get(list(self._metrics.keys())[0]).get('type') == REGRESSION:
                display_columns = display_columns + regression_metrics
            elif self._metrics.get(list(self._metrics.keys())[0]).get('type') == CLASSIFICATION:
                display_columns = display_columns + classification_metrics
            else:
                # This means that the first metrics is empty or did not have a type associated.
                # So, it is necessary to iterate over all the metrics to find a type
                idx = 1
                metric_keys = list(self._metrics.keys())
                length = len(metric_keys)

                # Break when a type is found
                while idx < length and self._metrics.get(metric_keys[idx]).get('type') is None:
                    idx = idx + 1

                # Check whether any metrics are present in the given experiments, else throw error
                assert_condition(condition=idx < length, source=self, message='No metrics found for given experiments')

                # For the identified type set the metrics
                if self._metrics.get(metric_keys[idx]).get('type') == REGRESSION:
                    display_columns = display_columns + regression_metrics

                else:
                    display_columns = display_columns + classification_metrics

        else:
            display_columns = display_columns + self._metrics_display

        data_report = []
        # For all runs that need to be displayed
        if self._display_run is None:
            return

        for run in self._display_run:

            # For all splits in a run
            for split in self._run_split_dict.get(run):

                metrics = self._metrics.get(split)
                if len(metrics) == 0:
                    continue

                # run_id split_id dataset
                data_row = dict()
                data_row['run'] = run
                data_row['split'] = split

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

                if metrics.get('type', None) == REGRESSION:
                    for metric in regression_metrics:
                        data_row[metric] = metrics.get(metric, '-')

                else:
                    for metric in classification_metrics:
                        data_row[metric] = metrics.get(metric, '-')

                curr_tuple = tuple()
                for col in display_columns:
                    curr_tuple = curr_tuple + tuple([str(data_row.get(col, '-'))])

                data_report.append(copy.deepcopy(curr_tuple))

        pd.options.display.max_rows = 999
        pd.options.display.max_columns = 15
        df = pd.DataFrame(data=data_report)

        if len(data_report) > 0 and len(data_report[0]) == len(display_columns):
            df.columns = display_columns

        # Update only those columns that are present in the data frame
        assert_condition(condition=metrics is not None, source=self, message='Could not compute metrics')
        if metrics.get('type', None) == REGRESSION:
            for metric in regression_metrics:
                if df.get(metric, None) is not None:
                    df[metric] = df[metric].astype(float)

        elif metrics.get('type', None) == CLASSIFICATION:
            for metric in classification_metrics:
                if df.get(metric, None) is not None:
                    df[metric] = df[metric].astype(float)

        # Remove empty columns in the data frame
        for name in df.columns:
            if all([str(df.get(name).values) == '-']):
                df = df.drop(name, axis=1)

        return df

    def get_experiment_directores(self):
        """
        Returns the list of the path of experiments currently available

        :return: List of path of experiments currently available
        """
        from pathlib import Path
        experiment_root_dir = os.path.join(str(Path.home()), '.pypadre/experiments')
        if not os.path.exists(experiment_root_dir):
            return None

        experiments_list = self.get_immediate_subdirectories(experiment_root_dir)
        return experiments_list

    def show_metrics(self):
        """
        This function combines all the other read and aggregate functions for ease of use
        :return:
        """
        self.read_run_directories()
        # From the run directory names, obtain the estimators and the parameters
        self.get_unique_estimators_parameter_names()
        # Read the JSON file objects from the .split folders
        self.read_split_metrics()
        # Get the results using Pandas data frame
        df = self.display_results()
        return df

    def add_experiments(self, experiments):
        """
        Adds a list of experiments for comparing the metrics
        :param experiments: List of experiment objects, paths or experiment names.
        :return: None
        """
        from pypadre.core.model.experiment import Experiment
        experiment_directory = []
        experiment_objects =[]
        experiment_names = []

        if isinstance(experiments, list):
            for experiment in experiments:

                trigger_event('EVENT_WARN', condition=isinstance(experiment, Experiment) or isinstance(experiment, str),
                              source=self,
                              message='Incorrect parameter in list. Only strings or experiment objects allowed')

                if isinstance(experiment, Experiment):
                    experiment_objects.append(experiment)

                elif isinstance(experiment, str):
                    if os.path.exists(experiment):
                        experiment_directory.append(experiment)

                    else:
                        experiment_names.append(experiment)

                else:
                    pass

            if len(experiment_objects) > 0:
                self.add_experiment_objects(experiment_objects)

            if len(experiment_directory) > 0:
                self.add_experiment_directory(experiment_directory)

            if len(experiment_names) > 0:
                self.add_experiment_by_name(experiment_names)

        else:
            assert_condition(condition=isinstance(experiments, Experiment) or isinstance(experiments, str), source=self,
                             message='Incorrect parameter type for comparing experiments. '
                                     'Only strings or experiment objects allowed')
            if isinstance(experiments, Experiment):
                self.add_experiment_objects(experiments)

            elif isinstance(experiments, str):
                if os.path.exists(experiments):
                    self.add_experiment_directory(experiments)

                else:
                    self.add_experiment_by_name(experiments)

            else:
                pass

    def add_experiment_objects(self, experiment_list):
        """
        Adds a list of experiment objects to the current list
        :param experiment_list: A list of experiment objects
        :return:
        """
        from pypadre.core.model.experiment import Experiment
        assert_condition(condition=isinstance(experiment_list, list) or isinstance(experiment_list, Experiment),
                         source=self, message='Incorrect input parameter type.')

        if isinstance(experiment_list, list):
            for experiment in experiment_list:
                assert_condition(condition=isinstance(experiment, Experiment), source=self,
                                 message='An object in the list is not of the type pypadre.core.experiment.Experiment')

        else:
            assert_condition(condition=isinstance(experiment_list, Experiment), source=self,
                             message='Incorrect parameter type')
            experiment_list = [experiment_list]

        if self._experiments is not None:
            self._experiments = self._experiments + experiment_list
        else:
            self._experiments = experiment_list

    def add_experiment_by_name(self, experiments):
        """
        Adds a list of experiments for computing metrics. These experiments should be present in the default config path
        :param experiments: List of experiment names for computing metrics
        :return:
        """
        assert_condition(condition=isinstance(experiments, list) or isinstance(experiments, str),
                         source=self, message='Incorrect input parameter type.')
        directory_list = []

        root_path = os.path.join(self._root_path, 'experiments')

        if isinstance(experiments, list):
            for directory in experiments:
                assert_condition(condition=isinstance(directory, str), source=self,
                                 message='An object in the list is not of the type string')

                # If only the name is given, add .ex to the end of experiment name.
                # Experiment directories are stored with .ex as suffix

                if directory[:-3] != '.ex':
                    path = os.path.join(root_path, directory+'.ex')

                else:
                    path = os.path.join(root_path, directory)

                # Check if such an experiment exists in the path
                if os.path.exists(path):
                    directory_list.append(path)

        else:
            assert_condition(condition=isinstance(experiments, str), source=self,
                             message='Incorrect parameter type')

            # If only the name is given, add .ex to the end of experiment name.
            # Experiment directories are stored with .ex as suffix
            if experiments[:-3] != '.ex':
                path = os.path.join(root_path, experiments+'.ex')

            else:
                path = os.path.join(root_path, experiments)

            # Check if such an experiment exists in the path
            if os.path.exists(path):
                directory_list.append(path)
            directory_list = [path]

        if self._dir_path is not None:
            self._dir_path = self._dir_path + directory_list
        else:
            self._dir_path = directory_list


    def add_experiment_directory(self, directory_list):
        """
        Adds a list of directory paths for computing metrics
        :param directory_list: List of directories
        :return:
        """

        assert_condition(condition=isinstance(directory_list, list) or isinstance(directory_list, str),
                         source=self, message='Incorrect input parameter type.')


        if isinstance(directory_list, list):
            for directory in directory_list:
                assert_condition(condition=isinstance(directory, str), source=self,
                                 message='An object in the list is not of the type string')

        else:
            assert_condition(condition=isinstance(directory_list, str), source=self,
                             message='Incorrect parameter type')
            directory_list = [directory_list]

        if self._dir_path is not None:
            self._dir_path = self._dir_path + directory_list
        else:
            self._dir_path = directory_list


class ComputeMetrics:
    """
    This class is responsible for computing the different metrics related to experiments
    """

    def compute_pr_curve(self, y_true, y_pred, probability, label=1):
        """
        This function returns a list of tuples having the p value, r value and confidence value
        :param y_true:
        :param y_pred:
        :param probability: A list of lists containing the probabilities for each class
        :param label: The label for which the precision recall curve has to be computed
        :return: a list of tuples containing (recall, precision, threshold)
        """
        assert_condition(condition=len(y_pred) == len(y_true) == len(probability), source=self,
                         message="Length mismatch between y_pred, y_true and probabilities")
        assert_condition(condition=isinstance(y_true, list) , source=self,
                         message='Function argument y_true is not a list')
        assert_condition(condition=isinstance(y_pred, list), source=self,
                         message='Function argument y_pred is not a list')
        assert_condition(condition=isinstance(probability, list), source=self,
                         message='Function argument probabilities is not a list')

        for idx in range(0, len(probability)):
            assert_condition(condition=isinstance(probability[idx], list), source=self,
                             message='An element at position {idx} is not a list in probabilities'.format(idx=idx+1))

        pr_value = []
        # Find out the number of classes
        num_classes = len(probability[0])
        if num_classes == 2:
            # If there are only two classes, find the maximum probabilities in each probability list
            max_probabilites = []
            for prob in probability:
                max_probabilites.append(max(prob))

            # Find the indices corresponding to the sorted probabilities list
            sorted_indices = sorted(range(len(max_probabilites)), key=lambda k: max_probabilites[k])

            # Compute precision and recall at each of those sorted indices
            for idx in range(0, len(sorted_indices)):
                # Compute precision and recall up to idx
                inner_idx = 0
                tp = 0
                tn = 0
                fp = 0
                fn = 0
                while inner_idx <= idx:
                    if y_pred[inner_idx] == 1 and y_true[inner_idx] == y_pred[inner_idx]:
                        tp += 1

                    elif y_pred[inner_idx] == 1 and y_true[inner_idx] == 0:
                        fp += 1

                    elif y_pred[inner_idx] == 0 and y_true[inner_idx] == 1:
                        fn += 1

                    elif y_pred[inner_idx] == 0 and y_true[inner_idx] == y_pred[inner_idx]:
                        tn += 1

                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                threshold = probability[idx]

                pr_value.append(tuple(recall, precision, threshold))

        else:

            # Modify the true and predictions to only include the indices that contain the label
            # as a true positive or a false positive

            modified_y_true = []
            modified_y_pred = []
            modified_probabilities = []

            tp = 0
            fp = 0
            tn = 0
            fn = 0

            for idx in range(0, len(y_pred)):
                if y_true[idx] == label or y_pred[idx] == label:
                    modified_y_true.append(y_true[idx])
                    modified_y_pred.append(y_pred[idx])
                    modified_probabilities.append(probability[idx])

                    # If there are only two classes, find the maximum probabilities in each probability list
                    max_probabilites = []
                    for prob in probability:
                        max_probabilites.append(max(prob))

                    # Find the indices corresponding to the sorted probabilities list
                    sorted_indices = sorted(range(len(max_probabilites)), key=lambda k: max_probabilites[k])

                    # Compute precision and recall at each of those sorted indices
                    for idx in range(0, len(sorted_indices)):
                        # Compute precision and recall up to idx
                        inner_idx = 0
                        tp = 0
                        tn = 0
                        fp = 0
                        fn = 0
                        while inner_idx <= idx:
                            if y_pred[inner_idx] == 1 and y_true[inner_idx] == y_pred[inner_idx]:
                                tp += 1

                            elif y_pred[inner_idx] == 1 and y_true[inner_idx] == 0:
                                fp += 1

                            elif y_pred[inner_idx] == 0 and y_true[inner_idx] == 1:
                                fn += 1

                            elif y_pred[inner_idx] == 0 and y_true[inner_idx] == y_pred[inner_idx]:
                                tn += 1

                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn)
                        threshold = probability[idx]

                        pr_value.append(tuple(recall, precision, threshold))

    def pr_curve(self, input, label=1):
        """
        This function extracts the splits from the experiment and obtains the precision recall curves for every split
        :param experiment: The path to an experiment/run/split directory or a experiment object
        :param label: Label for which precision recall curve is to be obtained in case of multiple labels, default: 1
        :return: A list of dictionaries containing the list of tuples having the pr curve values
        """

        # If only a split is given then, set default value 0 to experiment and run in dictionary, similarly for runs
        # Output will be a nested dictionary of experiment as outermost key, runs as the first level of nesting,
        # and splits as the second level of nesting

        pass

    def validate_input_parameters(self, input, label):
        """
        validates the input parameters to the function
        :param input: Can be a path to an experiment, run or split directory or an experiment, run or split object
        :param label: Label for which the pr curve is to be calculated for multi label classification
        :return: None
        """
        from pypadre.core.model.experiment import Experiment
        from pypadre.core.model.run import Run
        from pypadre.core.model.sklearnworkflow import SKLearnWorkflow
        from copy import deepcopy

        # Check if input type is correct
        flag = (isinstance(input, str) or isinstance(input, Experiment) or isinstance(input, Run) or \
               isinstance(input, SKLearnWorkflow)) and isinstance(label, int)

        assert_condition(condition=flag==True, source=self, message='Incorrect input types')

        flag = True
        if isinstance(input, str):
            flag = os.path.exists(input)
        assert_condition(condition=flag, source=self, message='Path {path} does not exist'.format(path=input))

        experiment = dict()

        # Check for results.json inside the split directory if a path is given
        if isinstance(input, str):
            # Identify if the path is an experiment path or run path or a split path
            if input[-2:] == 'ex':
                # Find all sub directories that are runs
                run_dir = self.get_valid_run_directories()

                assert_condition(condition=len(run_dir) > 0, source=self,
                                 message='Experiment directory does not contain any valid run directories')

                # Find all sub directories of runs that are splits
                split_directories = []
                for directory in run_dir:
                    split_directories = self.get_valid_split_directories()
                    # Get the UUID of the run
                    run_id = run_dir[:-4].split(sep='/')[-1]

                    # If there are valid splits add it to the run
                    if split_directories is not None:
                        experiment[run_id] = deepcopy(split_directories)

                # If there are no valid runs then raise an exception
                assert_condition(condition=len(experiment.keys()) > 0, source=self,
                                 message='Run directories do not contain any valid split directories')

                # Check if there is at least one results file
                results_files = []
                for directory in split_directories:
                    if os.path.exists(os.path.join(directory, 'results.json')):
                        results_files.append(directory)

            elif input[-3:] == 'run':
                # Find all split directories and check if results.json is present
                pass

            elif input[-5:] == 'split':
                # Check if results.json is present
                pass

            else:
                assert_condition(condition=False, source=self, message='Given directory path {path} does not correspond '
                                                                       'to an experiment, run or split '
                                                                       'directory'.format(path=input))


        elif isinstance(input, Experiment):
            # Check if results variable is not empty for runs and splits within
            pass

        elif isinstance(input, Run):
            # Check if results variable is not empty for all splits within
            pass

        elif isinstance(input, SKLearnWorkflow):
            # Check if results have all necessary values
            pass

        else:
            assert_condition(condition=False, source=self, message='Incorrect input parameter type given')

    def get_immediate_subdirectories(self, dir_path):
        """
        Gets the immediate subdirectories present in the dir_path.

        :param dir_path: The current experiment directory to be checked.

        :return: A list of run directories present in the experiment directory.
        """
        return [os.path.join(dir_path, name) for name in os.listdir(dir_path)
                if os.path.isdir(os.path.join(dir_path, name))]

    def get_valid_run_directories(self, exp_dir):
        """
        Returns the valid run directories when the path to an experiment directory is given
        :param exp_dir: Path to the experiment directory
        :return: List of valid run directories
        """
        sub_directories = self.get_immediate_subdirectories(exp_dir)
        run_dir = []
        for directory in sub_directories:
            if input[-3:] == 'run':
                run_dir.append(directory)

        return run_dir

    def get_valid_split_directories(self, run_dir):
        """
        Returns the valid split directories when an experiment directory is given as input
        :param run_dir: Path to a run directory
        :return: List containing valid split directories
        """
        split_directories = []
        for directory in run_dir:
            if input[-5:] == 'split':
                split_directories.append(directory)

        return split_directories