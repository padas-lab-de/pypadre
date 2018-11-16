"""
This Class helps is creating experiments that can be run by the Experiment class.
The class also performs the necessary validations on the experiment parameters.
The class abstracts the acquisition methods of the datasets too. The main aim of the class is to wrap the experiments
and enable the execution of multiple experiment sequentially. It also enables the execution of a single experiment
on multiple datasets.
"""
import ast
import importlib
import numpy as np
from copy import deepcopy
from padre.base import default_logger
from padre.ds_import import load_sklearn_toys
from padre.visitors.mappings import name_mappings
from padre.visitors.mappings import supported_frameworks
from padre.experiment import Experiment


class ExperimentCreator:
    # Initialization of the experiments dict.
    # This dictionary contains all the experiments to be executed.
    # All components of the experiment are contained in this dictionary.
    _experiments = dict()

    # The workflow components contain all the possible components of the workflow.
    # The key is an easy to remember string, and the value is the object of the component.
    # The object is deep copied to form the new component
    _workflow_components = dict()

    # This contains the all the classifiers for checking whether
    # classifiers are given continuous data
    _classifier_list = []

    # All parameters of estimators are to be listed here
    # as a list under the key of the estimator name
    # The parameters will be entered as a list to the key
    # based on the estimator name
    _parameters = dict()

    _param_value_dict = dict()

    _param_types_dict = dict()

    _estimator_alternate_names = dict()

    _param_type_mappings = {
                            u'integer':int,
                            u'float':float,
                            u'boolean':ast.literal_eval,
                           }

    # Mapping of the parameter names to the actual variable names
    _param_implementation = dict()

    # All the locally available datasets are mapped to this list.
    _local_dataset = []

    # All the datasets to be executed for the experiment are stored here

    def __init__(self):
        """
        Initialization function of the helper class.
        This function currently manually initializes all values but it could be changed to
        reading the data from files at run time.
        """
        self._workflow_components = self.initialize_workflow_components()

        self._parameters, self._param_implementation, self._param_types_dict = self.initialize_estimator_parameters_implementation()

        self._local_dataset = self.initialize_dataset_names()

        self.populate_alternate_names()

    def typecast_variable(self, param, allowed_types):
        """
        Dynamically typecasts the param variable based on the allowed types provided
        :param param: parameter to be type cast
        :param allowed_types: types that the parameter can be typecast into
        :return: a value that has been typecast
        """

        if len(allowed_types) > 1:
            allowed_types[0] = allowed_types[0][1:]
            allowed_types[-1] = allowed_types[-1][:-1]

        val = None
        for curr_type in allowed_types:
            try:
                val = self._param_type_mappings.get(curr_type, None)(param)
                break
            except:
                continue

        return val

    def convert_param_string_to_dictionary(self, param):
        """
        Converts a string to equivalent parameters by dynamic type casting

        :param param: the string containing the parameter name and parameter values

        :return: a dictionary with the parameter name as key and parameter values as values
        TODO: Currently supports only integer, float, boolean. Need to expand and find a method for precedence of types
        """
        # String Format: principal component analysis.n_components:[4, 5, 6, 7, 10]
        param_dict = dict()
        estimator_param_dict = dict()
        # Separate each estimator and corresponding parameters
        estimator_params_list = (param.strip()).split(sep="|")
        for estimator_params in estimator_params_list:
            curr_params = dict()

            # Extract each estimator name and corresponding parameter list
            sep_idx = estimator_params.find('.')
            if sep_idx == -1:
                default_logger.warn(False, 'ExperimentCreator.set_param_values',
                                    'Missing separators.')
                continue

            estimator = estimator_params[:sep_idx]
            params = (estimator_params[sep_idx + 1:]).split(',')
            sep_idx = params[0].find(':')
            if sep_idx == -1:
                default_logger.warn(False, 'ExperimentCreator.set_param_values',
                                    'Missing separators.')
                continue

            param_name = params[0][:sep_idx].strip()
            params[0] = params[0][sep_idx + 1:].strip()
            sep_idx = params[0].find('[')
            if sep_idx == -1:
                default_logger.warn(False, 'ExperimentCreator.set_param_values',
                                    'Missing separators.')
                continue

            params[0] = params[0][sep_idx + 1:].strip()
            sep_idx = params[-1].find("]")
            if sep_idx == -1:
                default_logger.warn(False, 'ExperimentCreator.set_param_values',
                                    'Missing separators.')
                continue

            params[-1] = params[-1][:sep_idx].strip()
            # For each parameter convert to corresponding type
            # Parameters that cannot be converted are discarded
            converted_params = []
            for idx in range(0, len(params)):
                type_string = self._param_types_dict.get('.'.join([estimator, param_name]), None)
                if type_string is None:
                    continue

                possible_types = type_string.replace(" ", "").split(sep=',')
                val = self.typecast_variable(params[idx].strip(), possible_types)

                if val is not None:
                    converted_params.append(val)

            curr_params[param_name] = deepcopy(list(converted_params))

            # if it is a new parameter for the estimator
            if param_dict.get(estimator, None) is None:
                param_dict[estimator] = deepcopy(curr_params)

            else:
                params_dict = param_dict.get(estimator)
                params_dict.update(deepcopy(curr_params))
                param_dict[estimator] = params_dict

        return param_dict

    def set_param_values(self, experiment_name=None, param=None):
        """
        This function sets the parameters for estimators in a single experiment

        :param experiment_name: The name of the experiment where the parameters need to be set
        :param param_dict: The estimator,parameter dictionary which specifies the parameters for the estimator

        :return: None
        """
        param_dict = None
        if param is None:
            default_logger.warn(False, 'ExperimentCreator.set_param_values',
                                'Missing parameter value')
            return None

        if isinstance(param, str):
            param_dict = self.convert_param_string_to_dictionary(param)

        elif isinstance(param, dict):
            param_dict = param


        if experiment_name is None:
            default_logger.warn(False, 'ExperimentCreator.set_param_values',
                                 'Missing experiment name when setting param values')
            return None

        if param_dict is None:
            default_logger.warn(False, 'ExperimentCreator.set_param_values', 'Missing dictionary argument')
            return None

        # Convert the names of the parameters to the actual described names in the mapping file
        param_dict = self.convert_alternate_estimator_names(params_dict=param_dict)

        self._param_value_dict[experiment_name] = self.validate_parameters(param_dict)

    def get_param_values(self, experiment_name=None):
        """
        This function displays all the parameters stored for a single experiment

        :param experiment_name: The name of the experiment whose parameters are to be examined

        :return: A string containing the estimators.parameters:[values]
        """

        estimator_params = self._param_value_dict.get(experiment_name)
        if estimator_params is None:
            return 'Experiment does not have any parameters set yet.'

        param_string = ''
        for estimator in estimator_params:

            param_dict = estimator_params.get(estimator)
            for param_name in param_dict:
                param_string += (estimator + '.')
                param_string += (param_name + ':')
                param_string += ''.join(str(param_dict.get(param_name)))
                param_string += "|"

        return param_string[:-1]

    def set_parameters(self, estimator, estimator_name, param_val_dict):
        """
        This function sets the parameters of the estimators and classifiers created
        in the pipeline. The param_val_dict contains the name of the parameter and
        the value to be set for that parameter.
        The parameters are looked up from the parameters dictionary.

        :parameter estimator: The estimator whose parameters have to be modified.
        :parameter estimator_name: The name of the estimator to look up whether,
                                   the parameters are present in that estimator.
        :parameter param_val_dict: The dictionary containing the parameter names
                                   and their values.

        :return If successful, the modified estimator
                Else, None.
        """
        if estimator is None:
            default_logger.error(False, 'ExperimentCreator.set_parameters',
                                 ''.join([estimator_name + ' does not exist in the workflow']))
            return None
        available_params = self._parameters.get(estimator_name)
        for param in param_val_dict:
            if param not in available_params:
                default_logger.error(False, 'ExperimentCreator.set_parameters',
                                     ''.join([param + ' is not present for estimator ' + estimator_name]))
            else:
                actual_param_name = self._param_implementation.get('.'.join([estimator_name, param]))
                estimator.set_params(**{actual_param_name: param_val_dict.get(param)})

        return estimator

    def validate_parameters(self, param_value_dict):
        """
        The function validates the parameters for each estimator and returns the validated parameters.
        The parameter names are changed to the actual parameter variable names to be used within the experiment class

        :param param_value_dict: The parameters and their corresponding values

        :return: A dictionary of the validated parameters
        """
        validated_param_dict = dict()
        for estimator_name in param_value_dict:
            # Check whether the estimator is available
            if self._workflow_components.get(estimator_name) is not None:
                # Check whether the params are available for the estimator
                parameters = param_value_dict.get(estimator_name)

                # If there are no parameters, then continue
                if parameters is None:
                    continue

                # The actual parameter names differ from the implementation names.
                # This function offers the flexibility of implementing the actual parameter names or their paths
                actual_parameter_names = []
                for param_name in self._parameters.get(estimator_name):
                    actual_parameter_names.append(
                        self._param_implementation.get('.'.join([estimator_name, param_name])))

                estimator_params = dict()
                for param in parameters:
                    if param in self._parameters.get(estimator_name):
                        actual_param_name = self._param_implementation.get('.'.join([estimator_name, param]))
                        estimator_params[actual_param_name] = parameters.get(param)

                    elif param in actual_parameter_names:
                        estimator_params[param] = parameters.get(param)

                    else:
                        default_logger.warn(False, 'ExperimentCreator.validate_parameters',
                                            ''.join([param, ' not present in list for estimator:', estimator_name]))

                if len(estimator_params) > 0:
                    validated_param_dict[estimator_name] = deepcopy(estimator_params)

            else:
                default_logger.warn(False, 'ExperimentCreator.validate_parameters',
                                    ''.join([estimator_name, ' not present in list']))

        if len(validated_param_dict) > 0:
            return validated_param_dict

        else:
            return None

    def validate_pipeline(self, pipeline):
        """
        This function checks whether each component in the pipeline has a fit and fit_transform attribute.
        The last estimator can be none and if it is not none, has to implement a fit attribute.

        :param pipeline: The pipeline of estimators to be creates

        :return: If successful True,
        """
        transformers = pipeline[:-1]
        estimator = pipeline[-1]

        for transformer in transformers:
            if (not (hasattr(transformer[1], "fit") or hasattr(transformer[1], "fit_transform")) or not
               hasattr(transformer[1], "transform")):
                default_logger.warn(False, 'ExperimentCreator.validate_pipeline',
                                     "All intermediate steps should implement fit "
                                     "and fit_transform or the transform function")
                return False

        if estimator is not None and not (hasattr(estimator[1], "fit")):
            default_logger.warn(False, 'ExperimentCreator.validate_pipeline',
                                 ''.join(["Estimator:" + estimator[0] + " does not have attribute fit"]))
            return False
        return True

    def create_test_pipeline(self, estimator_list, param_value_dict=None):
        """
        This function creates the pipeline for the experiment.
        The function looks up the name of the estimator passed and then, deep copies the estimator to the pipeline.

        :param estimator_list: A list of strings containing all estimators to be used
                               in the pipeline in exactly the order to be used
        :param param_value_dict: This contains the parameters for each of the estimators used
                                 The dict is a nested dictionary with the
                                 outer-most key being the estimator name
                                 The inner key is the parameter name

        :return: If successful, the Pipeline containing the estimators
                 Else, None
        """
        from sklearn.pipeline import Pipeline
        estimators = []

        if estimator_list is None:
            return None

        # If the params dict is not none, check whether any alternate estimator names are given and convert the
        # alternate estimator names to actual estimator names
        if param_value_dict is not None:
            name_updated_params = self.convert_alternate_estimator_names(param_value_dict)

        for estimator_name in estimator_list:
            if self._workflow_components.get(estimator_name, None) is None and \
                    self._estimator_alternate_names.get(str(estimator_name).upper(), None) is None:
                default_logger.error(False, 'ExperimentCreator.create_test_pipleline',
                                     ''.join([estimator_name + ' not present in list']))
                return None

            actual_estimator_name = estimator_name
            if self._estimator_alternate_names.get(str(estimator_name).upper(), None) is not None:
                actual_estimator_name = self._estimator_alternate_names.get(str(estimator_name).upper())

            # Copy of the estimator because the estimator object is mutable
            estimator = self.get_estimator_object(actual_estimator_name)
            estimators.append((actual_estimator_name, estimator))
            if param_value_dict is not None and \
                    name_updated_params.get(actual_estimator_name) is not None:
                self.set_parameters(estimator, actual_estimator_name, name_updated_params.get(actual_estimator_name))

        # Check if the created estimators are valid
        if not self.validate_pipeline(estimators):
            return False

        return Pipeline(estimators)

    def get_estimator_object(self, estimator):
        """
        This function instantiates a estimator from the estimator name

        :param estimator: Name of the estimator

        :return: An object of the estimator
        """

        if estimator is None:
            return None

        path = self._workflow_components.get(estimator)
        split_idx = path.rfind('.')
        import_path = path[:split_idx]
        class_name = path[split_idx+1:]
        module = importlib.import_module(import_path)
        class_ = getattr(module, class_name)
        obj = class_()
        return deepcopy(obj)

    def create_experiment(self, name, description,
                          dataset_list=None, workflow=None,
                          backend=None, params=None):
        """
        This function adds an experiment to the dictionary.

        :param name: Name of the experiment. It should be unique for this set of experiments
        :param description: The description of the experiment
        :param dataset_list: The names of the datasets to be used for the experiment
        :param workflow: The scikit pipeline to be used for the experiment.
        :param backend: The backend of the experiment
        :param params: Parameters for the estimator, optional.

        :return: None
        """
        import numpy as np
        if name is None:
            default_logger.warn(False, 'ExperimentCreator.create_experiment',
                                'Experiment name is missing, a name will be generated by the system')

        if description is None or \
                workflow is None or backend is None or workflow is False:
            if description is None:
                default_logger.error(False, 'ExperimentCreator.create_experiment',
                                     ''.join(['Description is missing for experiment:', name]))
            if backend is None:
                default_logger.error(False, 'ExperimentCreator.get_local_dataset',
                                     ''.join(['Backend is missing for experiment:', name]))
            return None

        # If the name of the dataset is passed, the get the local dataset and replace it
        if isinstance(dataset_list, str) or isinstance(dataset_list, list):
            if isinstance(dataset_list, str):
                dataset_list = [dataset_list]
            # TODO: Check whether a classification pipeline is assigned to a regression dataset

            # Experiment name should be unique
            if self._experiments.get(name, None) is None:
                data_dict = dict()
                data_dict['description'] = description
                data_dict['dataset'] = dataset_list
                data_dict['workflow'] = workflow
                data_dict['backend'] = backend
                self._experiments[name] = data_dict
                if params is not None:
                    self._param_value_dict[name] = self.validate_parameters(params)
                    data_dict['params'] = self.validate_parameters(params)
                default_logger.log('ExperimentCreator.create_experiment',
                                   ''.join([name, ' created successfully!']))

            else:
                default_logger.error(False, 'ExperimentCreator.create_experiment', 'Error creating experiment')
                if self._experiments.get(name, None) is not None:
                    default_logger.error(False, 'ExperimentCreator.create_experiment',
                                         ''.join(['Experiment name: ', name,
                                                  ' already present. Experiment name should be unique']))

    def get_local_dataset(self, name=None):
        """
        This function returns the dataset from pypadre.
        This done by using the pre-defined names of the datasets defined in _local_dataset

        TODO: Datasets need to be loaded dynamically

        :param name: The name of the dataset

        :return: If successful, the dataset
                 Else, None
        """
        if name is None:
            default_logger.error(False, 'ExperimentCreator.get_local_dataset', 'Dataset name is empty')
            return None
        if name in self._local_dataset:
            return [i for i in load_sklearn_toys()][self._local_dataset.index(name)]
        else:
            default_logger.error(False, 'ExperimentCreator.get_local_dataset', name + ' Local Dataset not found')
            return None

    def get_dataset_names(self):
        """
        This function returns all the names of available datasets

        :return: A list of dataset names
        """
        return list(self._local_dataset)

    def get_estimators(self):
        """
        Gets all the available estimators

        :return: List of names of the estimators
        """
        return list(self._workflow_components.keys())

    def get_estimator_params(self, estimator_name):
        """
        Gets the parameters corresponding to an estimator

        :param estimator_name: Name of the estimator whose params are to be retrieved

        :return: List of parameters available to that estimator
        """
        return self._parameters.get(estimator_name, None)

    def initialize_workflow_components(self):
        """
        This function returns all the workflow components along with their object initialized with default parameters

        :return: A dictionary containing all the available estimators
        """
        components = dict()
        for estimator in name_mappings:
            components[estimator] = name_mappings.get(estimator).get('implementation').get('scikit-learn')

        return components

    def initialize_estimator_parameters_implementation(self):
        """
        The function returns the parameters corresponding to each estimator in use

        TODO: Write checks for library implementations

        :return: Dictionary containing estimator and the corresponding parameters with its implementation
        """

        estimator_params = dict()
        param_implementation_dict = dict()
        param_types_dict = dict()
        for estimator in name_mappings:
            param_list = []
            param_list_dict = name_mappings.get(estimator).get('hyper_parameters').get('model_parameters')
            for param in param_list_dict:
                param_list.append(param.get('name'))
                for framework in supported_frameworks:
                    if param.get(framework, None) is not None:
                        implementation = param.get(framework).get('path', None)
                        if implementation is not None:
                            param_implementation_dict['.'.join([estimator, param.get('name')])] = implementation

                param_types_dict['.'.join([estimator, param.get('name')])] = \
                    param.get('kind_of_value')
            estimator_params[estimator] = deepcopy(param_list)

        return estimator_params, deepcopy(param_implementation_dict), deepcopy(param_types_dict)

    def initialize_dataset_names(self):
        """
        The function returns all the datasets currently available to the user. It can be from the server and also the
        local datasets availabe

        TODO: Dynamically populate the list of datasets

        :return: List of names of available datasets
        """
        dataset_names = ['Boston_House_Prices',
                         'Breast_Cancer',
                         'Diabetes',
                         'Digits',
                         'Iris',
                         'Linnerrud']

        return dataset_names

    def execute_experiments(self):
        """
        This function runs all the created experiments from the experiment dictionary

        :return: None
        """
        import pprint
        if self._experiments is None:
            return

        for experiment in self._experiments:
            flag = True
            dataset = deepcopy(self._experiments.get(experiment).get('dataset', None))
            if dataset is None:
                default_logger.warn(False, 'Experiment_creator.execute_experiments',
                                    'Dataset is not present for the experiment. Experiment ' + experiment + 'is ignored.')
                continue

            if len(dataset) == 1:

                # Get the data from the dataset name
                data = self.get_local_dataset(dataset[0])

                # Classifiers cannot work on continuous data and rejected as experiments.
                if not np.all(np.mod(data.targets(), 1) == 0):
                    workflow = self._experiments.get(experiment).get('workflow', None)
                    for estimator in workflow.named_steps:
                        if name_mappings.get(estimator).get('type', None) == 'Classification':
                            flag = False
                            default_logger.warn(False, 'ExperimentCreator.do_experiments',
                                                ''.join(
                                                    ['Estimator ', estimator, ' cannot work on continuous data.'
                                                                              'This dataset will be disregarded']))
                            break

                # If the pipeline consists of classification estimators working on continous data return
                if not flag:
                    continue

                # If there is only one dataset defined for the experiment execute the experiment with the dataset
                ex = Experiment(name=experiment,
                                description=self._experiments.get(experiment).get('description'),
                                dataset=data,
                                workflow=self._experiments.get(experiment).get('workflow', None),
                                backend=self._experiments.get(experiment).get('backend', None),
                                strategy=self._experiments.get(experiment).get('strategy', 'random'))

                conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline
                pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
                ex.grid_search(parameters=self._param_value_dict.get(experiment))

            else:
                # If there are multiple datasets defined for the experiment execute the experiment for each dataset
                datasets = dataset
                for dataset in datasets:

                    # If such an experiment does not exist, discard
                    if self._experiments.get(experiment, None) is None:
                        continue

                    # If the dataset does not exist, discard
                    if datasets is None:
                        continue

                    flag = True
                    desc = ''.join([self._experiments.get(experiment).get('description'), 'with dataset ', dataset])
                    data = self.get_local_dataset(dataset)

                    if data is None:
                        continue

                    # Classifiers cannot work on continuous data and rejected as experiments.
                    if not np.all(np.mod(data.targets(), 1) == 0):
                        workflow = self._experiments.get(experiment).get('workflow', None)
                        for estimator in workflow.named_steps:
                            if name_mappings.get(estimator).get('type', None) == 'Classification':
                                flag = False
                                default_logger.warn(False, 'ExperimentCreator.execute_experiments',
                                                    ''.join(
                                                        ['Estimator ', estimator, ' cannot work on continuous data.'
                                                                                  'This dataset will be disregarded']))

                    # If a classification estimator tries to work on continous data disregard it
                    if not flag:
                        continue

                    message = 'Executing experiment ' + experiment + ' for dataset' + dataset
                    default_logger.log('ExperimentCreator.do_experiments', message)

                    ex = Experiment(name=''.join([experiment, '(', dataset, ')']),
                                    description=desc,
                                    dataset=data,
                                    workflow=self._experiments.get(experiment).get('workflow', None),
                                    backend=self._experiments.get(experiment).get('backend', None),
                                    strategy=self._experiments.get(experiment).get('strategy', 'random'))
                    conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline

                    pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
                    ex.grid_search(parameters=self._param_value_dict.get(experiment))

    def do_experiments(self, experiment_datasets=None):
        """
        This function runs the same experiment over multiple datasets

        :param experiment_datasets: A dictionary containing the datasets to be run for each experiment

        :return: None
        """

        import pprint
        import numpy as np

        if experiment_datasets is None:
            return None

        for experiment in experiment_datasets:

            # If such an experiment does not exist, discard
            if self._experiments.get(experiment, None) is None:
                continue

            datasets = experiment_datasets.get(experiment, None)
            # If the dataset does not exist, discard
            if datasets is None:
                continue

            for dataset in datasets:
                flag = True
                desc = ''.join([self._experiments.get(experiment).get('description'), 'with dataset ', dataset])
                data = self.get_local_dataset(dataset)

                if data is None:
                    continue

                # Classifiers cannot work on continuous data and rejected as experiments.
                if not np.all(np.mod(data.targets(), 1) == 0):
                    workflow = self._experiments.get(experiment).get('workflow', None)
                    for estimator in workflow.named_steps:
                        if name_mappings.get(estimator).get('type', None) == 'Classification':
                            flag = False
                            default_logger.warn(False, 'ExperimentCreator.do_experiments',
                                                 ''.join(['Estimator ', estimator, ' cannot work on continuous data.'
                                                                                   'This dataset will be disregarded']))

                # If a classification estimator tries to work on continous data disregard it
                if not flag:
                    continue

                message = 'Executing experiment ' + experiment + ' for dataset ' + dataset
                default_logger.log('ExperimentCreator.do_experiments', message)

                ex = Experiment(name=''.join([experiment, '(', dataset, ')']),
                                description=desc,
                                dataset=data,
                                workflow=self._experiments.get(experiment).get('workflow', None),
                                backend=self._experiments.get(experiment).get('backend', None),
                                strategy=self._experiments.get(experiment).get('strategy', 'random'))
                conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline

                pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
                ex.grid_search(parameters=self._param_value_dict.get(experiment))

    def populate_alternate_names(self):
        """
        This function populates the alternate names of estimators from name_mappings

        :return: None
        """

        for estimator in name_mappings:
            estimator_params = name_mappings.get(estimator)

            other_names = estimator_params.get('other_names', None)

            for name in other_names:
                self._estimator_alternate_names[str(name).upper()] = estimator

    def convert_alternate_estimator_names(self, params_dict):
        """
        This function converts the alternate parameter names to actual parameter names to pass to the experiment class

        :param params_dict: A dictionary containing the parameters of the estimator

        :return: A dictionary containing the parameter with the key as the actual estimator name
        """

        modified_params_dict = dict()

        for estimator_name in params_dict:

            curr_params = params_dict.get(estimator_name)

            if self._workflow_components.get(estimator_name, None) is not None:
                modified_params_dict[estimator_name] = deepcopy(curr_params)

            # User has used an alternate name of the estimator
            elif self._workflow_components.get(estimator_name, None) is None and \
                    self._estimator_alternate_names.get(str(estimator_name).upper(), None) is not None:

                actual_estimator_name = self._estimator_alternate_names.get(str(estimator_name).upper(), None)
                modified_params_dict[actual_estimator_name] = deepcopy(curr_params)

        return deepcopy(modified_params_dict)

    def parse_config_file(self, filename):
        """
        The function parses a JSON file which contains the necessary parameters for creating experiments

        :param filename: Path of the JSON file

        :return: True if successfully parsed
                 False if file does not exist
        """

        import os
        import json
        from padre.app import pypadre

        if not (os.path.exists(filename)):
            default_logger.warn(False, 'ExperimentCreator.parse_config_file', f"File does not exist {filename}. Return")
            return False

        # Load the experiments structure from the file
        with open(filename, 'r') as f:
            experiments = json.loads(f.read())

        for experiment in experiments:
            # Iterate and obtain the parameters of all the experiments
            exp_params = experiments.get(experiment)
            if exp_params is None:
                continue

            name = exp_params.get('name', None)
            description = exp_params.get('description', None)
            pipeline = exp_params.get('workflow', None)
            dataset = exp_params.get('dataset', None)
            backend = exp_params.get('backend', 'default')
            params = exp_params.get('params', None)

            if backend == 'default':
                backend = pypadre.file_repository.experiments

            # Create the pipeline and if it is not possible move to next experiment
            workflow = self.create_test_pipeline(pipeline)
            if workflow is None:
                default_logger.warn(False, 'ExperimentCreator.parse_config_file',
                                    'Workflow  based workflow was not created')
                continue

            self.create_experiment(name=name, description=description,workflow=workflow, dataset_list=dataset,
                                   backend=backend, params=params)

        return True

    def setExperimentDatasets(self, params):
        """
        This function sets datasets for the corresponding experiment

        :param params: A dictionary containing experiment names
        and the corresponding datasets to be used for that experiment

        :return: None
        """

        for experiment in params:
            if self._experiments.get(experiment) is None:
                continue

            dataset = params.get(experiment)

            if isinstance(dataset, str):
                self._experiments[experiment] = [dataset]

            elif isinstance(dataset, list):
                # TODO: Verify all datasets before copying
                self._experiments[experiment]['dataset'] = deepcopy(dataset)

    def createExperimentList(self):
        """
        This function creates a list of experiments that can be executed by an external function

        :return: A list of dictionaries containing all the experiments to be executed
        """
        experiments_list = []
        for experiment_name in self._experiments:
            experiment_dict = dict()
            experiment = self._experiments.get(experiment_name)
            dataset = experiment.get('dataset')
            if len(dataset) == 1:
                experiment_dict['name'] = experiment_name
                experiment_dict['description'] = experiment.get('description')
                experiment_dict['workflow'] = experiment.get('workflow')
                experiment_dict['dataset'] = dataset[0]
                experiment_dict['backend'] = experiment.get('backend')
                experiment_dict['params'] = experiment.get('params')
                experiments_list.append(deepcopy(experiment_dict))

            elif len(dataset) > 1:
                datasets = dataset
                for dataset in datasets:
                    experiment_dict['name'] = ''.join([experiment_name, '(', dataset, ')'])
                    experiment_dict['description'] = ''.join([experiment.get('description'), ' with dataset ', dataset])
                    experiment_dict['workflow'] = experiment.get('workflow')
                    experiment_dict['dataset'] = dataset
                    experiment_dict['backend'] = experiment.get('backend')
                    experiment_dict['params'] = experiment.get('params')
                    experiments_list.append(deepcopy(experiment_dict))

        return deepcopy(experiments_list)

    @property
    def experiments(self):
        return self._experiments

    @property
    def experiment_names(self):
        return list(self._experiments.keys())

    @property
    def components(self):
        return list(self._workflow_components.keys())
