import itertools
import platform
import pypadre.core.visitors.parameter

from collections import OrderedDict
from pypadre.eventhandler import trigger_event, assert_condition
from pypadre.base import MetadataEntity
from pypadre.core.datasets import Dataset
from pypadre.core.validatetraintestsplits import ValidateTrainTestSplits
from pypadre.core.sklearnworkflow import SKLearnWorkflow
from pypadre.core.run import Run
from pypadre.core.custom_split import split_obj
from pypadre.core.visitors.mappings import name_mappings, alternate_name_mappings, supported_frameworks
####################################################################################################################
#  Module Private Functions and Classes
####################################################################################################################


def _sklearn_runner():
    pass


def _is_sklearn_pipeline(pipeline):
    """
    checks whether pipeline is a sklearn pipeline
    :param pipeline:
    :return:
    """
    # we do checks via strings, not isinstance in order to avoid a dependency on sklearn
    return type(pipeline).__name__ == 'Pipeline' and type(pipeline).__module__ == 'sklearn.pipeline'


class Experiment(MetadataEntity):
    """
    Experiment class covering functionality for executing and evaluating machine learning experiments.
    It is determined by a pipeline which is evaluated over a dataset with several configuration.
    A run applies one configuration over the data, which can be splitted in several sub-runs on different dataset parts
    in order to get reliable statistical estimates.

    An experiment requires:
    1. a pipeline / workflow. A workflow implements `fit`, `infer` and `transform` methods, comparable to sklearn.
    Currently, we only support sklearn pipelines, which are wrapped by the SKLearnWorkflow
    <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>,
    i.e. a list of (name, class) tuples, where the class implements:
       - a `fit` function (parameters need to be defined)
       - a `infer` function in case of supervised prediction (parameters need to be defined)
       - a `transform`function in case of feature space transformers (parameters need to be defined)

    2. a dataset. An experiment is always tight to one dataset which is the main dataset for running the experiment.
       Future work should allow to include auxiliary resources, but currently we only support one dataset.
    3. metadata describing different aspects of the workflow.
      - the splitting strategy (see Splitter)
      - hyperparameter ranges
      - output control etc.


    Experiment Metadata:
    ====================

    All metadata provided to the experiment will be stored along the experiment description. However, the following
    properties will gain special purpose for an experiment:
    - task - determines the task achieved by a experiment (e.g. classification, regression, metric learning etc.)
    - name - determines the name of an experiment
    - id - determines the repository id of an experiment (might be equal to the name, if the name is also the id)
    - description - determines the description of an experiment
    - domain - determines the application domain

    Parameters required:
    ===================
    The following parameters need to be set in the constructor or via annotations
    - dataset : pypadre.datasets.Dataset
    - workflow: either a pypadre.experiment.Workflow object or a SKLearn Pipeline

    Options supported:
    ==================
    - stdout={True|False} logs event messages to default_logger. Default = True
    - keep_splits={True|False} if true, all split data for every run is kept (including the model, split inidices and training data)
                               are kept in memory. If false, no split data is kept
    - keep_runs={True|False} if true, all rund data (i.e. scores) will be kept in memory. If false, no split run data is not kept
    - n_runs = int  number of runs to conduct. todo: needs to be extended with hyperparameter search

    TODO:
    - Queuing mode
    - Searching Hyperparameter Space

    """

    _id = None
    _metadata = None

    def __init__(self,
                 **options):
        # Validate input types
        self.validate_input_parameters(options=options)


        self._dataset = options.pop("dataset", None)
        assert_condition(condition=self._dataset is not None, source=self, message="Dataset cannot be none")
        assert_condition(condition=isinstance(self._dataset, Dataset),
                         source=self, message='Experiment dataset is not of type Dataset')
        # we need to store the dataset_id in the metadata. otherwise, this information might be lost during storage
        options["dataset_id"] = self._dataset.id
        # todo workflow semantic not clear. Fit and infer is fine, but we need someting for transform
        workflow = options.pop("workflow", None)
        self._stdout = options.get("stdout", True)
        self._keep_runs = options.get("keep_runs", False) or options.get("keep_splits", False)
        self._runs = []
        self._run_split_dict = OrderedDict()
        self._sk_learn_stepwise = options.get("sk_learn_stepwise", False)
        self._set_workflow(workflow)
        self._last_run = None
        self._validation_obj = options.get('validation', None)
        self._results = []
        self._metrics = []
        self._hyperparameters = []
        self._experiment_configuration = None

        # If a preprocessing step is required to be executed on the whole dataset, add the workflow
        self._preprocessed_workflow = options.pop('preprocessing', None)

        # Deep copy the modified dataset to the variable after preprocessing
        self._preprocessed_dataset = None

        # Set the flag after preprocessing is complete
        self._preprocessed = False

        split_obj.function_pointer = options.pop('function', None)
        super().__init__(options.pop("ex_id", None), **options)

        if self._validation_obj is None or not hasattr(self._validation_obj, 'validate'):
            self._validation_obj = ValidateTrainTestSplits()

        self._fill_sys_info()

    def _fill_sys_info(self):
        # TODO: Implement the gathering of system information as dynamic code
        # TODO: Remove hard coded strings.
        # This function collects all system related info in a dictionary
        sys_info = dict()
        sys_info["processor"] = platform.processor()
        sys_info["machine"] = platform.machine()
        sys_info["system"] = platform.system()
        sys_info["platform"] = platform.platform()
        sys_info["platform_version"] = platform.version()
        sys_info["node_name"] = platform.node()
        sys_info["python_version"] = platform.python_version()
        self._metadata["sys_info"] = sys_info

    def _set_workflow(self, w):
        if _is_sklearn_pipeline(w):
            self._workflow = SKLearnWorkflow(w, self._sk_learn_stepwise)
        else:
            self._workflow = w

    @property
    def run_split_dict(self):
        return self._run_split_dict

    @run_split_dict.setter
    def run_split_dict(self, run_split_dict):
        self._run_split_dict = run_split_dict

    @property
    def dataset(self):
        if not self._preprocessed:
            return self._dataset
        else:
            return self._preprocessed_dataset

    @dataset.setter
    def dataset(self, ds):
        self._dataset = ds

    @property
    def workflow(self):
        return self._workflow

    @property
    def validate(self):
        return self._validation_obj

    @property
    def experiment_configuration(self):
        return self._experiment_configuration

    @experiment_configuration.setter
    def experiment_configuration(self, configuration):
        self._experiment_configuration = configuration

    @workflow.setter
    def workflow(self, w):
        self._set_workflow(w)

    def configuration(self):
        return self._workflow.configuration()

    def hyperparameters(self):
        """
        returns the hyperparameters per pipeline element as dict from the extracted configruation
        :return:
        """
        # todo only list the hyperparameters and, if available, the potential value ranges
        # todo experiment.hyperparameters() should deliver a json serialisable object.
        # make it as close to the http implementation as possible
        params = dict()
        steps = self.configuration()[0]["steps"]
        # Params is a dictionary of hyper parameters where the key is the zero-indexed step number
        # The traverse_dict function traverses the dictionary in a recursive fashion and replaces
        # any instance of <class 'pypadre.core.visitors.parameter.Parameter'> type to a sub-dictionary of
        # value and attribute. This allows the dictionary to be JSON serializable
        for idx, step in enumerate(steps):
            params["".join(["Step_", str(idx)])] = self.traverse_dict(dict(step))
        return params

    def set_hyperparameters(self, hyperparameters):
        # todo placeholder as loading an experiment should include loading hyperparameters.
        # However, in sklearn, the hyperparameters are defined via the pipeline. As long as
        # we do not integrate a second framework, we do not need the mechanism
        pass

    @property
    def workflow(self):
        return self._workflow

    def run(self, append_runs: bool = False):
        """
        runs the experiment
        :param append_runs: If true, the runs will be appended if the experiment exists already.
        Otherwise, the experiment will be deleted
        :return:
        """
        from copy import deepcopy

        # Update metadata with version details of packages used in the workflow
        self.update_experiment_metadata_with_workflow()

        # todo allow split wise execution of the individual workflow steps. some kind of reproduction / debugging mode
        # which gives access to one split, the model of the split etc.
        # todo allow to append runs for experiments
        # register experiment through logger
        # self.logger.log_start_experiment(self, append_runs)
        trigger_event('EVENT_START_EXPERIMENT', experiment=self, append_runs=self._keep_runs)

        r = Run(self, self._workflow, **dict(self._metadata))
        r.do_splits()
        if self._keep_runs:
            self._runs.append(r)
        self._results.append(deepcopy(r.results))
        self._metrics.append(deepcopy(r.metrics))
        self._hyperparameters = [(deepcopy(r.hyperparameters))]
        self._last_run = r
        self._run_split_dict[str(r.id) + '.run'] = r.split_ids
        trigger_event('EVENT_STOP_EXPERIMENT', experiment=self)

    def execute(self, parameters=None):
        """
        This function searches a grid of the parameter combinations given into the function
        :param parameters: A nested dictionary, where the outermost key is the estimator name and
        the second level key is the parameter name, and the value is a list of possible parameters
        :return: None
        """

        from copy import deepcopy

        assert_condition(condition=parameters is None or isinstance(parameters, dict),
                         source=self,
                         message='Incorrect parameter type to the execute function')

        if self._preprocessed_workflow is not None:
            self.preprocess()

        if parameters is None:
            self._experiment_configuration = self.create_experiment_configuration_dict(params=None, single_run=True)
            self.run()

            # Fire event
            trigger_event('EVENT_PUT_EXPERIMENT_CONFIGURATION', experiment=self)
            return

        # Update metadata with version details of packages used in the workflow
        self.update_experiment_metadata_with_workflow()

        # Generate every possible combination of the provided hyper parameters.
        workflow = self._workflow
        master_list = []
        params_list = []

        # Fire event
        trigger_event('EVENT_START_EXPERIMENT', experiment=self, append_runs=self._keep_runs)

        for estimator in parameters:
            param_dict = parameters.get(estimator)
            assert_condition(condition=isinstance(param_dict, dict),
                             source=self,
                             message='Parameter dictionary is not of type dictionary for estimator:' + estimator)
            for params in param_dict:
                # Append only the parameters to create a master list
                master_list.append(param_dict.get(params))

                # Append the estimator name followed by the parameter to create a ordered list.
                # Ordering of estimator.parameter corresponds to the value in the resultant grid tuple
                params_list.append(''.join([estimator, '.', params]))
        grid = itertools.product(*master_list)

        self._experiment_configuration = self.create_experiment_configuration_dict(params=parameters, single_run=False)

        # Fire event
        trigger_event('EVENT_PUT_EXPERIMENT_CONFIGURATION', experiment=self)

        # Get the total number of iterations
        grid_size = 1
        for idx in range(0, len(master_list)):
            grid_size *= len(master_list[idx])

        # Starting index
        curr_executing_index = 1

        # For each tuple in the combination create a run
        for element in grid:
            trigger_event('EVENT_LOG_EVENT', source=self,
                          message="Executing grid " + str(curr_executing_index) + '/' + str(grid_size))
            trigger_event('EVENT_LOG_RUN_PROGRESS', curr_value=curr_executing_index, limit=str(grid_size), phase='start')
            # Get all the parameters to be used on set_param
            for param, idx in zip(params_list, range(0, len(params_list))):
                split_params = param.split(sep='.')
                estimator = workflow._pipeline.named_steps.get(split_params[0])

                if estimator is None:
                    assert_condition(condition=estimator is not None, source=self,
                                  message=f"Estimator {split_params[0]} is not present in the pipeline")
                    break

                estimator.set_params(**{split_params[1]: element[idx]})

            r = Run(self, workflow, **dict(self._metadata))
            r.do_splits()
            self._run_split_dict[str(r.id)+'.run'] = r.split_ids
            if self._keep_runs:
                self._runs.append(r)

            self._results.append(deepcopy(r.results))
            self._metrics.append(deepcopy(r.metrics))
            self._last_run = r
            self._hyperparameters.append(deepcopy(r.hyperparameters))

            trigger_event('EVENT_LOG_RUN_PROGRESS', curr_value=curr_executing_index, limit=str(grid_size),
                          phase='stop')
            curr_executing_index += 1

        # Fire event
        trigger_event('EVENT_STOP_EXPERIMENT', experiment=self)

    def preprocess(self):
        """
        Runs the preprocessing pipeline and populates the preprocessed dataset
        :return: None
        """
        from copy import deepcopy

        # Preprocess the data
        preprocessed_data = self._preprocessed_workflow.fit_transform(self.dataset.features(), self.dataset.targets)
        # Copy the dataset so that metadata and attributes remain consistent
        self._preprocessed_dataset = deepcopy(self.dataset)

        # Replace the data by concatenating with the targets
        self._preprocessed_dataset.replace_data(preprocessed_data)
        # Set flag
        self._preprocessed = True

    def create_experiment_configuration_dict(self, params=None, single_run=False):
        """
        This function creates a dictionary that can be written as a JSON file for replicating the experiments.

        :param params: The parameters for the estimators that make up the grid
        :param single_run: If the execution is done for a single run

        :return: Experiment dictionary containing the pipeline, backend, parameters etc
        """
        from copy import deepcopy

        name = self.name
        description = self.metadata.get('description', None)
        strategy = self.metadata.get('strategy', None)
        dataset = self.dataset.name
        workflow = list(self.workflow.pipeline.named_steps.keys())

        complete_experiment_dict = dict()

        experiment_dict = dict()
        experiment_dict['name'] = name
        experiment_dict['description'] = description
        experiment_dict['strategy'] = strategy
        experiment_dict['dataset'] = dataset
        experiment_dict['workflow'] = workflow

        # If there is a preprocessing pipeline, add it to the configuration
        if self._preprocessed is True:
            preprocessing_workflow = list(self._preprocessed_workflow.named_steps.keys())
            experiment_dict['preprocessing'] = preprocessing_workflow

        if single_run is True:
            estimator_dict = dict()
            # All the parameters of the estimators need to be filled into the params dictionary
            estimators = self.workflow.pipeline.named_steps
            for estimator in estimators:

                obj_params = estimators.get(estimator).get_params()
                estimator_name = estimator
                if name_mappings.get(estimator, None) is None:
                    estimator_name = alternate_name_mappings.get(estimator)

                params_list = name_mappings.get(estimator_name).get('hyper_parameters').get('model_parameters')
                params = estimators.get(estimator).get_params()
                param_dict = dict()
                for param in params_list:
                    for framework in supported_frameworks:
                        if param.get(framework, None) is not None:
                            break
                    param_name = param.get(framework).get('path')
                    param_dict[param_name] = obj_params.get(param_name)

                estimator_dict[estimator] = deepcopy(param_dict)

            experiment_dict['params'] = estimator_dict

        else:
            # Only those parameters that are passed to the grid search need to be filled
            experiment_dict['params'] = params

        complete_experiment_dict[name] = deepcopy(experiment_dict)

        return complete_experiment_dict

    @property
    def runs(self):
        if self._runs is not None:
            return self._runs
        else:
            # load splits from backend.
            raise NotImplemented()

    @property
    def last_run(self):
        return self._last_run

    @property
    def results(self):
        return self._results

    @property
    def metrics(self):
        return self._metrics

    @property
    def hyperparameters_combinations(self):
        return self._hyperparameters

    @property
    def requires_preprocessing(self):
        return self._preprocessed

    @property
    def preprocessing_workflow(self):
        return self._preprocessed_workflow

    def __str__(self):
        s = []
        if self.id is not None:
            s.append("id:" + str(self.id))
        if self.name is not None and self.name != self.id:
            s.append("name:" + str(self.name))
        if len(s) == 0:
            return str(super())
        else:
            return "Experiment<" + ";".join(s) + ">"

    def traverse_dict(self, dictionary=None):
        """
        This function traverses a Nested dictionary structure such as the
        parameter dictionary obtained from hyperparameters()
        The aim of this function is to convert the param objects to
        JSON serializable form. The <class 'pypadre.core.visitors.parameter.Parameter'> type
        is used to store the base values. This function changes the type to basic JSON
        serializable data types.

        :param dictionary: The dictionary containing all the parameters of the pipeline

        :return: A JSON serializable object containing the parameter tree
        """

        if dictionary is None:
            return

        for key in dictionary:
            if isinstance(dictionary[key], pypadre.core.visitors.parameter.Parameter):
                dictionary[key] = {'value': dictionary[key].value,
                                   'attributes': dictionary[key].attributes}

            elif isinstance(dictionary[key], dict):
                self.traverse_dict(dictionary[key])

        return dictionary

    def update_experiment_metadata_with_workflow(self):
        """
        This function updates the experiment's metadata with details of the different modules used in the pipeline and
        the corresponding version number of the modules.

        :return: None
        """
        import importlib

        modules = list()
        modules.append('pypadre')
        module_version_info = dict()

        estimators = self._workflow._pipeline.named_steps
        # Iterate through the entire pipeline and find the unique modules
        for estimator in estimators:
            obj = estimators.get(estimator, None)

            # If the estimator has module attribute, get the name of the module
            if estimator is not None and hasattr(obj, "__module__"):
                # module name would be of the form sklearn.utils.
                # Split out only the first part from the module
                module_name = obj.__module__
                split_idx = module_name.find('.')
                # If it is a padre package, it may have its own package version, so keep the full path
                if module_name[:split_idx] != 'padre':
                    module_name = module_name[:split_idx]

                # Add the module name if it is not present
                if module_name not in modules:
                    modules.append(module_name)

        # Obtain the version information of all the modules present in the list
        for module in modules:
            module_ = importlib.import_module(module)
            if hasattr(module_, "__version__"):
                module_version_info[module] = module_.__version__

        self.metadata['versions'] = module_version_info

    def validate_input_parameters(self, options):
        """
        This function validates all the parameters given to the experiment constructor
        :param options: Dictionary containing the parameters given to the constructor of the class
        :return: True if successful validation of parameters, False if not
        """
        from pypadre.core.visitors.mappings import name_mappings, alternate_name_mappings
        import numpy as np

        assert_condition(condition=options.get('workflow', None) is not None, source=self,
                         message="Workflow cannot be none")
        assert_condition(condition=options.get('description', None) is not None, source=self,
                         message="Description cannot be none")
        assert_condition(condition=isinstance(options.get("keep_runs", True), bool), source=self,
                         message='keep_runs parameter has to be of type bool')
        assert_condition(condition=isinstance(options.get("keep_splits", True), bool), source=self,
                         message='keep_splits parameter has to be of type bool')
        assert_condition(condition=isinstance(options.get('sk_learn_stepwise', False), bool), source=self,
                         message='keep_splits parameter has to be of type bool')
        assert_condition(condition=hasattr(options.get('workflow', dict()), 'fit') is True, source=self,
                         message='Workflow does not have a fit function')
        assert_condition(condition=isinstance(options.get('name', 'noname'), str) or options.get('name') is None,
                         source=self, message='Experiment name should be of type string')
        assert_condition(condition=isinstance(options.get('description', 'noname'), str),
                         source=self, message='Experiment description should be of type string')
        assert_condition(condition=options.get('dataset', None) is not None, source=self,
                         message="Dataset cannot be none")
        assert_condition(condition=isinstance(options.get('dataset', dict()), Dataset),
                         source=self, message='Experiment dataset is not of type Dataset')
        assert_condition(condition=options.get('preprocessing', None) is None or hasattr(options.get('preprocessing',
                                                                                                     dict()),
                                                                                         'fit_transform') is True,
                         source=self,
                         message='Preprocessing workflow does not have a fit_transform function')

        # Check if all estimator names are present in the name mappings
        workflow = options.get('workflow')
        for estimator in workflow.named_steps:
            assert_condition(condition=name_mappings.get(estimator, None) is not None or
                             alternate_name_mappings.get(estimator, None) is not None,
                             source=self,
                             message='Estimator {estimator} not present in name mappings or '
                                     'alternate name mappings'.format(estimator=estimator))

        # Check if dataset has targets, and if supervised learning is used, then throw an error
        if options.get('dataset').targets() is None:
            workflow = options.get('workflow')
            for estimator in workflow.named_steps:
                actual_estimator_name = estimator
                if name_mappings.get(estimator, None) is None:
                    actual_estimator_name = alternate_name_mappings.get(estimator)
                assert_condition(
                    condition=name_mappings.get(actual_estimator_name).get('type', None) not in
                              ['Classification', 'Regression'],
                    source=self, message='Dataset without targets cannot be used for supervised learning')

        # Check if regression data is assigned to a classification estimator
        if not np.all(np.mod(options.get('dataset').targets(), 1) == 0):
            workflow = options.get('workflow')
            for estimator in workflow.named_steps:
                actual_estimator_name = estimator
                if name_mappings.get(estimator, None) is None:
                    actual_estimator_name = alternate_name_mappings.get(estimator)
                assert_condition(condition=name_mappings.get(actual_estimator_name).get('type', None) != 'Classification',
                                 source=self, message='Classifier cannot be trained on regression data')

