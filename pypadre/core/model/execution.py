from collections import OrderedDict

from pypadre import Experiment
from pypadre.pod.base import MetadataEntity
from pypadre.core.model.run import Run
from pypadre.pod.eventhandler import assert_condition, trigger_event
from pypadre.pod.printing.tablefyable import Tablefyable
from pypadre.pod.validation import Validateable


class Execution(Validateable, MetadataEntity, Tablefyable):
    """ A execution should save data about the running env and the version of the code on which it was run """

    _metadata = None

    @classmethod
    def tablefy_register_columns(cls):
        # Add entries for tablefyable
        cls._tablefy_register_columns({'hash': 'hash', 'cmd': 'cmd'})

    def __init__(self, experiment: Experiment, codehash=None, command=None, **options):
        metadata = {"id": codehash, **options, "command": command, "codehash": codehash}
        Validateable.__init__(self, schema_resource_name="execution.json", **metadata)
        MetadataEntity.__init__(self, **metadata)
        Tablefyable.__init__(self)

        # Validate input types
        parameters = options.pop('parameters', None)
        preparameters = options.pop('preparameters', None)
        single_run = options.pop('single_run', True)
        single_transformation = options.pop('single_transformation', True)
        self._keep_runs = options.get('keep_runs', True)

        self.validate_input_parameters(experiment=experiment, options=options)

        self._experiment = experiment
        self._runs = []
        self._hash = codehash
        self._cmd = command

        if self.name is None:
            self.name = self._hash

        self._experiment_configuration = self.\
            create_experiment_configuration_dict(params=parameters,
                                                 preprocessing_params=preparameters,
                                                 single_transformation=single_run,
                                                 single_run=single_transformation)

        self._results = []
        self._metrics = []
        self._run_split_dict = OrderedDict()

    @property
    def hash(self):
        return self._hash

    @property
    def results(self):
        return self._results

    @property
    def metrics(self):
        return self._metrics

    @property
    def cmd(self):
        return self._cmd

    @property
    def config(self):
        return self._experiment_configuration

    @property
    def experiment(self):
        return self._experiment

    def validate_input_parameters(self, experiment, options):
        from pypadre.core.model.experiment import Experiment
        assert_condition(condition=experiment is not None, source=self,
                         message="Experiment cannot be None")
        assert_condition(condition=isinstance(experiment, Experiment), source=self,
                         message="Parameter experiment is not an object of padre.core.Experiment")

    def execute(self, parameters, preprocessed_workflow):
        """
        This function searches a grid of the parameter combinations given into the function
        :param parameters: A nested dictionary, where the outermost key is the estimator name and
        the second level key is the parameter name, and the value is a list of possible parameters
        :return: None
        """

        from copy import deepcopy
        import itertools

        assert_condition(condition=parameters is None or isinstance(parameters, dict),
                         source=self,
                         message='Incorrect parameter type to the execute function')

        if preprocessed_workflow is not None:
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
            trigger_event('EVENT_LOG_RUN_PROGRESS', curr_value=curr_executing_index, limit=str(grid_size),
                          phase='start')
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
            self._run_split_dict[str(r.id) + '.run'] = r.split_ids
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

    def run(self, append_runs: bool = False):
        """
        runs the experiment
        :param append_runs: If true, the runs will be appended if the experiment exists already.
        Otherwise, the experiment will be deleted
        :return:
        """
        from copy import deepcopy

        # Update metadata with version details of packages used in the workflow
        self.experiment.update_experiment_metadata_with_workflow()

        # todo allow split wise execution of the individual workflow steps. some kind of reproduction / debugging mode
        # which gives access to one split, the model of the split etc.
        # todo allow to append runs for experiments
        # register experiment through logger
        # self.logger.log_start_experiment(self, append_runs)

        r = Run(self, self.experiment.workflow, **dict(self._metadata))
        r.do_splits()
        if self._keep_runs:
            self._runs.append(r)
        self._results.append(deepcopy(r.results))
        self._metrics.append(deepcopy(r.metrics))
        self._hyperparameters = [(deepcopy(r.hyperparameters))]
        self._last_run = r
        self._run_split_dict[str(r.id) + '.run'] = r.split_ids

    def create_experiment_configuration_dict(self, params=None, preprocessing_params=None, single_run=False,
                                             single_transformation=False):
        """
        This function creates a dictionary that can be written as a JSON file for replicating the experiments.

        :param params: The parameters for the estimators that make up the grid for the main workflow
        :param preprocessing_params: The parameters for the transformers that make up the grid for the preprocessing workflow
        :param single_run: If the execution is done for a single run
        :param single_transformation: If the dataset has a single transformation

        :return: Experiment dictionary containing the pipeline, backend, parameters etc
        """
        from copy import deepcopy
        from pypadre.core.visitors.mappings import name_mappings, alternate_name_mappings, supported_frameworks

        name = self.experiment.name
        description = self.experiment.metadata.get('description', None)
        strategy = self.experiment.metadata.get('strategy', None)
        dataset = self.experiment.dataset.name
        workflow = list(self.experiment.workflow.pipeline.named_steps.keys())

        complete_experiment_dict = dict()

        experiment_dict = dict()
        experiment_dict['name'] = name
        experiment_dict['description'] = description
        experiment_dict['strategy'] = strategy
        experiment_dict['dataset'] = dataset
        experiment_dict['workflow'] = workflow

        # If there is a preprocessing pipeline, add it to the configuration
        if self.experiment.requires_preprocessing:
            preprocessing_workflow = list(self.experiment._preprocessed_workflow.named_steps.keys())
            experiment_dict['preprocessing'] = preprocessing_workflow

        if single_run is True:
            estimator_dict = dict()
            # All the parameters of the estimators need to be filled into the params dictionary
            estimators = self.experiment.workflow.pipeline.named_steps
            for estimator in estimators:

                obj_params = estimators.get(estimator).get_params()
                estimator_name = estimator
                if name_mappings.get(estimator, None) is None:
                    estimator_name = alternate_name_mappings.get(str(estimator))
                    if estimator_name is None:
                        estimator_name = alternate_name_mappings.get(str(estimator).lower())

                params_list = name_mappings.get(estimator_name).get('hyper_parameters').get('model_parameters')
                param_dict = dict()
                framework = dict()
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

        if self.experiment.requires_preprocessing:

            if single_transformation is True:
                transformer_dict = dict()
                # All the parameters of the transformers need to be filled into the params dictionary
                transformers = self.experiment.preprocessing_workflow.named_steps
                for transformer in transformers:
                    obj_params = transformers.get(transformer).get_params()
                    transformer_name = transformer
                    if name_mappings.get(transformer, None) is None:
                        transformer_name = alternate_name_mappings.get(str(transformer).lower())

                    params_list = name_mappings.get(transformer_name).get('hyper_parameters').get('model_parameters')
                    param_dict = dict()
                    framework = dict()
                    for param in params_list:
                        for framework in supported_frameworks:
                            if param.get(framework, None) is not None:
                                break
                        param_name = param.get(framework).get('path')
                        param_dict[param_name] = obj_params.get(param_name)

                    transformer_dict[transformer] = deepcopy(param_dict)

                experiment_dict['preprocessing_params'] = transformer_dict

            else:
                # Only those parameters that are passed to the grid search need to be filled
                experiment_dict['preprocessing_params'] = preprocessing_params

        complete_experiment_dict[name] = deepcopy(experiment_dict)

        return complete_experiment_dict
