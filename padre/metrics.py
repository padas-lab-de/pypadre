"""
Classes for managing the metrics related to experiments.
The metrics can be visualized as pandas dataframes or the metrics
could be recalculated.
"""

import copy
import json
import numpy as np
import os
import pandas as pd
from padre.base import default_logger


class ReevaluationMetrics:
    """
    This class reevaluates the metrics from the results.json file present in the split folder.
    The user should be able to specify the required metrics and those additional metrics will be computed, if
    the functions for those metrics are available
    """
    def __init__(self, dir_path=None, file_path=None):
        if dir_path is not None:
            self._dir_path = dir_path

        if file_path is not None:
            self._file_path = file_path

        self._split_dir = []

    def get_immediate_subdirectories(self, dir_path):
        """
        Gets the immediate subdirectories present in the dir_path
        :param dir_path: The current experiment directory to be checked
        :return: A list of run directories present in the experiment directory
        """
        return [os.path.join(dir_path, name) for name in os.listdir(dir_path)
                if os.path.isdir(os.path.join(dir_path, name))]

    def get_split_directories(self):

        for experiment_path in self._dir_path:
            run_dir_list = self.get_immediate_subdirectories(experiment_path)

            for run_path in run_dir_list:
                split_dir_list = self.get_immediate_subdirectories(run_path)

                for split_dir in split_dir_list:
                    if os.path.exists(os.path.join(split_dir, 'results.json')):
                        self._split_dir.append(split_dir)

    def recompute_metrics(self):

        for split_path in self._split_dir:

            # Load the hyperparameters.json file from the run directory
            with open(os.path.join(split_path, "results.json"), 'r') as f:
                results = json.loads(f.read())

            prediction_type = results.get('type', None)
            dataset = results.get('dataset', '-')
            metrics = None
            if prediction_type is None:
                default_logger.error('No prediction type present in ' + split_path)
                continue

            elif prediction_type == 'regression':
                metrics = self.compute_regression_metrics(results)

            elif prediction_type == 'classification':
                metrics = self.compute_classification_metrics(results)

            if metrics is not None:
                metrics['type'] = prediction_type
                metrics['dataset'] = dataset
                with open(os.path.join(split_path, "metrics.json"), 'w') as f:
                    f.write(json.dumps(metrics))

            else:
                default_logger.error('Error reading the results.json file in ' + split_path)

    def compute_regression_metrics(self, results=None):

        if results is None:
            default_logger.error('Results is None')
            return

        y_true = results.get('truth', None)
        y_predicted = results.get('predicted', None)

        if y_true is None or y_predicted is None:
            default_logger.error('Truth or Predicted values missing in results.json')
            return None

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

        if results is None:
            default_logger.error('Results is empty')
            return None

        y_true = results.get('truth', None)
        y_predicted = results.get('predicted', None)
        option = results.get('average', 'macro')

        if y_true is None or y_predicted is None:
            default_logger.error('Truth or Predicted values missing in results.json')
            return None

        confusion_matrix = self.compute_confusion_matrix(predicted=y_predicted,
                                                         truth=y_true)

        if confusion_matrix is None:
            default_logger.error('Could not compute confusion matrix')
            return None

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
        This was done as a general purpose implementation of the confusion_matrix
        :param predicted: The predicted values of the confusion matrix
        :param truth: The truth values of the confusion matrix
        :return: The confusion matrix
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
                default_logger.error('Empty run directory')
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

            # This dictionary contains the default values for each estimator,
            # so that only those parameters that are different across estimators
            # need to be displayed
            estimator_default_values = dict()

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

            self._display_run = copy.deepcopy(list(self._run_split_dict.keys()))

        def compute_results(self):
            """
            Computes the collected data as a Pandas data frame
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
                if self._metrics.get(list(self._metrics.keys())[0]).get('type') == 'regression':
                    display_columns = display_columns + regression_metrics
                else:
                    display_columns = display_columns + classification_metrics

            else:
                display_columns = display_columns + self._metrics_display

            data_report = []
            # For all runs that need to be displayed
            if self._display_run is None:
                default_logger.error('No runs to be displayed')
                return

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
            if len(data_report) > 0 and len(data_report[0]) == len(display_columns):
                df.columns = display_columns
            return df

        def analyze_runs(self, query=None, metrics=None, options=None):
            """
            This function would return a pandas data frame based on the query
            and the metrics required
            :param query: Initially, this would be a list of directories to be evaluated upon
            :param metrics: The metrics to be displayed
            :param options: Any other option possible, like micro averaged, macro averaged etc
            :return: A pandas data frame containing the results of the query
            """
            if query is None:
                default_logger.error('Function called with empty query')
                return

            row_id_set = set()

            for element in query:

                if str(element).lower() == 'all':
                    self._display_run = copy.deepcopy(self._run_split_dict)
                    return

                if self._run_estimators.get(element, None) is not None:
                    row_id_set = row_id_set.union(set(self._run_estimators.get(element, None)))

                elif self._param_values_run_id.get(element, None) is not None:
                    row_id_set = row_id_set.union(set(self._param_values_run_id.get(element, None)))

                else:
                    # Query not present within lists
                    default_logger.error(element + 'not present')

            self._display_run = copy.deepcopy(list(row_id_set))

            self._metrics_display = []
            if metrics is not None:
                for metric in metrics:
                    self._metrics_display.append(metric)

        def get_unique_estimator_names(self):
            """
            This function returns the unique estimators among the runs
            :return: A list of the names of unique estimators
            """
            return copy.deepcopy(list(self._unique_estimators.keys()))

        def get_estimator_param_values(self, estimator_name, selected_params=None, return_all_values=False):
            """
            This function returns all the parameter values present among the runs for a particular estimator
            :param estimator_name: he name of the estimator whose differing parameters are to be fetched
            :param selected_params: Displays the parameter values of the specified parameters only
            :param return_all_values: whether a list of the default values are also to be returned
            :return: List of param values, None if the estimator is not present
            """
            params_list = dict()
            params = self._unique_estimators.get(estimator_name, None)
            if params is None:
                default_logger.error('No parameters obtained for estimator :' + estimator_name)
                return None

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

