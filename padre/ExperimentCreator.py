"""
This class creates experiments that can be run by the Experiment class
All the validation mechanisms on the parameters to the Experiment class are implemented here
This class also provides interfaces to find the available datasets, available estimators,
and the estimator parameters
"""
import copy
import importlib
from padre.base import default_logger
from padre.ds_import import load_sklearn_toys
from padre.visitors.mappings import type_mappings, name_mappings
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
    # TODO: This should mapped dynamically
    _parameters = dict()

    _param_value_dict = dict()

    # All the locally available datasets are mapped to this list.
    _local_dataset = []

    def __init__(self):
        """
        Initialization function of the helper class.
        This function currently manually initializes all values but it could be changed to
        reading the data from files at run time.
        TODO: Make this function read all values dynamically from a file
        """

        self._workflow_components = self.initialize_workflow_components()

        self._classifiers = self.intialize_classifiers()

        self._parameters = self.initialize_estimator_parameters()

        self._local_dataset = self.initialize_dataset_names()

    def set_param_values(self, experiment_name=None, param_dict=None):

        if experiment_name is None:
            default_logger.error('Experiment_creator.set_param_values',
                                 'Missing experiment name when setting param values')

        if param_dict is None:
            default_logger.error('Experiment_creator.set_param_values','Missing dictionary argument')

        self._param_value_dict[experiment_name] = copy.deepcopy(param_dict)


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
            default_logger.error(False, 'Experiment_creator.set_parameters',
                                 ''.join([estimator_name + ' does not exist in the workflow']))
            return None
        available_params = self._parameters.get(estimator_name)
        for param in param_val_dict:
            if param not in available_params:
                default_logger.error(False, 'Experiment_creator.set_parameters',
                                     ''.join([param + ' is not present for estimator ' + estimator_name]))
            else:
                estimator.set_params(**{param: param_val_dict.get(param)})

        return estimator

    def validate_pipeline(self, pipeline):
        """
        This function checks whether each component in the pipeline has a
        fit and fit_transform attribute or transform attribute. The last estimator
        can be none and if it is not none, has to implement a fit attribute.
        :param pipeline: The pipeline of estimators to be created.
        :return: If successful True
                 Else False
        """
        transformers = pipeline[:-1]
        estimator = pipeline[-1]

        for transformer in transformers:
            if (not (hasattr(transformer[1], "fit") or hasattr(transformer[1], "fit_transform")) or not
               hasattr(transformer[1], "transform")):
                default_logger.error(False, 'Experiment_creator.validate_pipeline',
                                     "All intermediate steps should implement fit "
                                     "and fit_transform or the transform function")
                return False

        if estimator is not None and not (hasattr(estimator[1], "fit")):
            default_logger.error(False, 'Experiment_creator.validate_pipeline',
                                 ''.join(["Estimator:" + estimator[0] + " does not have attribute fit"]))
            return False
        return True

    def create_test_pipeline(self, estimator_list, param_value_dict=None):
        """
        This function creates the pipeline for the experiment.
        The function looks up the name of the estimator passed and then
        deep copies the estimator to the pipeline.
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
        for estimator_name in estimator_list:
            if self._workflow_components.get(estimator_name) is None:
                default_logger.error(False, 'Experiment_creator.create_test_pipleline',
                                     ''.join([estimator_name + ' not present in list']))
                return None

            # Deep copy of the estimator because the estimator object is mutable
            estimator = self.get_estimator_object(estimator_name)
            estimators.append((estimator_name, estimator))
            if param_value_dict is not None and \
                    param_value_dict.get(estimator_name) is not None:
                self.set_parameters(estimator, estimator_name, param_value_dict.get(estimator_name))

        # Check if the created estimators are valid
        if not self.validate_pipeline(estimators):
            return False

        return Pipeline(estimators)

    def get_estimator_object(self, estimator):

        if estimator is None:
            return None

        path = self._workflow_components.get(estimator)
        split_idx = path.rfind('.')
        import_path = path[:split_idx]
        class_name = path[split_idx+1:]
        module = importlib.import_module(import_path)
        class_ = getattr(module, class_name)
        obj = class_()
        return copy.deepcopy(obj)


    def create_experiment(self, name, description,
                          dataset, workflow,
                          backend, params=None):
        """
        This function adds an experiment to the dictionary.
        :param name: Name of the experiment. It should be unique for this set of experiments
        :param description: The description of the experiment
        :param dataset: The dataset to be used for the experiment
        :param workflow: The scikit pipeline to be used for the experiment.
        :param backend: The backend of the experiment
        :param params: Parameters for the estimator, optional.
        :return: None
        """
        import numpy as np
        if name is None:
            default_logger.warn('Experiment name is missing, a name will be generated by the system')

        if description is None or \
                dataset is None or workflow is None or backend is None or workflow is False:
            if description is None:
                default_logger.error(False, 'Experiment_creator.create_experiment',
                                     ''.join(['Description is missing for experiment:', name]))
            if dataset is None:
                default_logger.error(False, 'Experiment_creator.create_experiment',
                                     ''.join(['Dataset is missing for experiment:', name]))
            if backend is None:
                default_logger.error(False, 'Experiment_creator.get_local_dataset',
                                     ''.join(['Backend is missing for experiment:', name]))
            return None

        # Classifiers cannot work on continuous data and rejected as experiments.
        if not np.all(np.mod(dataset.targets(), 1) == 0):
            for estimator in workflow.named_steps:
                if name_mappings.get(estimator).get('type', None) != 'Classification':
                    default_logger.error(False, 'Experiment_creator.create_experiment',
                                         ''.join(['Estimator ', estimator, ' cannot work on continuous data']))
                    return None

        # Experiment name should be unique
        if self._experiments.get(name, None) is None:
            data_dict = dict()
            data_dict['description'] = description
            data_dict['dataset'] = dataset
            data_dict['workflow'] = workflow
            data_dict['backend'] = backend
            self._experiments[name] = data_dict
            if params is not None:
                self._param_value_dict[name] = copy.deepcopy(params)
            default_logger.log('Experiment_creator.get_local_dataset',
                               ''.join(['Experiment Creator', 'Experiment:', name, ' created successfully']))

        else:
            default_logger.error(False, 'Error creating experiment')
            if self._experiments.get(name, None) is not None:
                default_logger.error(False, 'Experiment_creator.create_experiment',
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
            default_logger.error(False, 'Experiment_creator.get_local_dataset', 'Dataset name is empty')
            return None
        if name in self._local_dataset:
            return [i for i in load_sklearn_toys()][self._local_dataset.index(name)]
        else:
            default_logger.error(False, 'Experiment_creator.get_local_dataset', name + ' Local Dataset not found')
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

    def initialize_workflow_components(self):
        """
        This function returns all the workflow components along with their object initialized with default parameters
        :return: A dictionary containing all the available estimators
        """
        components = dict()
        for estimator in name_mappings:
            components[estimator] = name_mappings.get(estimator).get('implementation').get('scikit-learn')

        return components

    def intialize_classifiers(self):
        """
        This function returns a list of estimators that are classifiers.
        This is required because classifiers cannot work on continuous data
        TODO: Dynamically populate the list
        :return: List containing the names of all the classifiers
        """
        return ['SVC']

    def initialize_estimator_parameters(self):
        """
        The function returns the parameters corresponding to each estimator in use
        TODO: Dynamically populate the list
        :return: Dictionary containing estimator and parameters
        """
        estimator_params = {'SVC': ['probability', 'C', 'degree'],
                            'LSA': ['n_components'],
                            'isomap': ['n_neighbors', 'n_components'],
                            'lle': ['n_neighbors', 'n_components', 'method'],
                            'SVR': ['kernel', 'degree', 'gamma', 'C']}

        return estimator_params

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
        :return:
        """
        import pprint
        if self._experiments is None:
            return

        for experiment in self._experiments:
            ex = Experiment(name=experiment,
                            description=self._experiments.get(experiment).get('description'),
                            dataset=self._experiments.get(experiment).get('dataset', None),
                            workflow=self._experiments.get(experiment).get('workflow', None),
                            backend=self._experiments.get(experiment).get('backend', None),
                            strategy=self._experiments.get(experiment).get('strategy', 'random'))

            conf = ex.configuration()  # configuration, which has been automatically extracted from the pipeline
            pprint.pprint(ex.hyperparameters())  # get and print hyperparameters
            ex.grid_search(parameters=self._param_value_dict.get(experiment))


    @property
    def experiments(self):
        return self._experiments
