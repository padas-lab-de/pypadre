# TODO DELETE AS SOON AS TRANSFERED INTO NEW APP CLASSES

"""
Padre app as single point of interaction.

Defaults:

- The default configuration is provided under `.padre.cfg` in the user home directory


Architecture of the module
++++++++++++++++++++++++++

- `PadreConfig` wraps the configuration for the app. It can read/write the config from a file (if provided)

"""

# todo merge with cli. cli should use app and app should be configurable via builder pattern and configuration files
import configparser
import inspect
import os
from functools import wraps
from typing import List, Union

from docutils.nodes import warning
from jsonschema import ValidationError

from pypadre.core.model.dataset.dataset import Dataset
from pypadre.core.model.generic.custom_code import _convert_path_to_code_object
from pypadre.core.model.pipeline.parameter_providers.parameters import FunctionParameterProvider
from pypadre.core.printing.tablefyable import Tablefyable
from pypadre.core.printing.util.print_util import to_table
from pypadre.core.util.utils import filter_nones
from pypadre.pod.app.base_app import IBaseApp
from pypadre.pod.app.code_app import CodeApp
from pypadre.pod.app.config.padre_config import PadreConfig
from pypadre.pod.app.dataset.dataset_app import DatasetApp
from pypadre.pod.app.metric_app import MetricApp
from pypadre.pod.app.project.computation_app import ComputationApp
from pypadre.pod.app.project.execution_app import ExecutionApp
from pypadre.pod.app.project.experiment_app import ExperimentApp
from pypadre.pod.app.project.pipeline_output_app import PipelineOutputApp
from pypadre.pod.app.project.project_app import ProjectApp
from pypadre.pod.app.project.run_app import RunApp
from pypadre.pod.app.project.split_app import SplitApp
from pypadre.pod.backend.file import PadreFileBackend
from pypadre.pod.backend.gitlab import PadreGitLabBackend
from pypadre.pod.backend.i_padre_backend import IPadreBackend


# logger = PadreLogger(app=None)
# add_logger(logger=logger)


class PadreAppFactory:

    @staticmethod
    def get(config=PadreConfig()):
        backends = PadreAppFactory._parse_backends(config)
        return PadreApp(backends=backends)

    @staticmethod
    def _parse_backends(config):
        _backends = config.get("backends", "GENERAL")

        backends = []
        for b in _backends:
            if 'base_url' in b:
                # TODO check for validity
                pass
                # backends.append(PadreHttpBackend(b))
            elif 'gitlab_url' in b:
                #TODO check for validity
                backends.append(PadreGitLabBackend(b))
            elif 'root_dir' in b:
                # TODO check for validity
                backends.append(PadreFileBackend(b))
            else:
                raise ValidationError('{0} defined an invalid backend. Please provide either a http backend'
                                      ' or a local backend. (root_dir or base_url)'.format(b))
        return backends


class PadreApp(IBaseApp):

    # TODO metric algorithms should be passed for metric calculation. This should work a bit like on the server. Metrics themselves are plugins which are invoked by the reevaluater
    def __init__(self, printer=None, backends: List[IPadreBackend] = None):
        super().__init__()
        self._print = printer

        if backends is None:
            pass
            # TODO inform the user

        self._backends = backends

        # TODO Should each subApp really hold each backend? This may be convenient to code like this.
        #  self._logger = LoggingService(backends)
        self._dataset_app = DatasetApp(self,
                                       [backend.dataset for backend in backends] if backends is not None else None)
        self._project_app = ProjectApp(self,
                                       [backend.project for backend in backends] if backends is not None else None)
        self._experiment_app = ExperimentApp(self, [backend.experiment for backend in
                                                    backends] if backends is not None else None)
        self._execution_app = ExecutionApp(self, [backend.execution for backend in
                                                  backends] if backends is not None else None)
        self._run_app = RunApp(self, [backend.run for backend in backends] if backends is not None else None)
        self._split_app = SplitApp(self, [backend.split for backend in backends] if backends is not None else None)
        self._computation_app = ComputationApp(self, [backend.computation for backend in
                                                      backends] if backends is not None else None)
        self._metric_app = MetricApp(self, [backend.metric for backend in backends] if backends is not None else None)
        self._code_app = CodeApp(self, [backend.code for backend in backends] if backends is not None else None)
        self._pipeline_output_app = PipelineOutputApp(self, [backend.pipeline_output for backend in
                                                             backends] if backends is not None else None)

    @property
    def backends(self):
        return self._backends

    @property
    def datasets(self):
        return self._dataset_app

    @property
    def projects(self):
        return self._project_app

    @property
    def experiments(self):
        return self._experiment_app

    @property
    def executions(self):
        return self._execution_app

    @property
    def runs(self):
        return self._run_app

    @property
    def splits(self):
        return self._split_app

    @property
    def computations(self):
        return self._computation_app

    @property
    def metrics(self):
        return self._metric_app

    @property
    def code(self):
        return self._code_app

    def workflow(self, *args, ptype, parameters=None, parameter_provider=None,
                 dataset: Union[Dataset, str], project_name=None, experiment_name=None,
                 project_description=None,
                 experiment_description=None, auto_main=True, **kwargs):
        """
        Decroator for functions that return a single workflow to be executed in an experiment with name exp_name
        :param exp_name: name of the experiment
        :param args: additional positional parameters to an experiment (replaces other positional parameters if longer)
        :param kwargs: kwarguments for experiments
        :return:
        """

        if parameters is None:
            parameters = {}

        def workflow_decorator(f_create_workflow):
            @wraps(f_create_workflow)
            def wrap_workflow(*args, **kwargs):
                # here the workflow gets called. We could add some logging etc. capability here, but i am not sure
                return f_create_workflow(*args, **kwargs)

            (filename, _, function_name, _, _) = inspect.getframeinfo(inspect.currentframe().f_back)
            creator = _convert_path_to_code_object(filename, function_name)

            if ptype is None:
                # TODO look up the class by parsing the mapping / looking at the return value of the function or something similar
                raise NotImplementedError()


            # TODO check pipeline type (where to put provider)
            if parameter_provider is not None:
                pipeline = ptype(pipeline_fn=wrap_workflow, parameter_provider=parameter_provider, creator=creator)
            else:
                pipeline = ptype(pipeline_fn=wrap_workflow, creator=creator)

            project = self.projects.get_by_name(project_name)
            if project is None:
                project = self.projects.create(
                    **filter_nones({"name": project_name, "description": project_description}), creator=creator)

            d = dataset if isinstance(dataset, Dataset) else self.datasets.get_by_name(dataset)
            experiment = self.experiments.create(
                **filter_nones({"name": experiment_name, "description": experiment_description}),
                project=project,
                pipeline=pipeline, dataset=d, creator=creator)
            if auto_main:
                return experiment.execute(parameters=parameters)
            else:
                if parameters:
                    warning("Parameters are given but experiment is not started directly. Parameters will be omitted. "
                            "You have to pass them on the execute call again.")
                return experiment

        return workflow_decorator

    def parameter_map(self):
        def parameter_decorator(f_create_parameters):
            @wraps(f_create_parameters)
            def wrap_parameters(*args, **kwargs):
                # here the parameter map gets called. We could add some logging etc. capability here,
                # but i am not sure
                return f_create_parameters(*args, **kwargs)
            return wrap_parameters()
        return parameter_decorator

    def parameter_provider(self, *args, **kwargs):
        def parameter_decorator(f_create_parameters):
            @wraps(f_create_parameters)
            def wrap_parameters(*args, **kwargs):
                # here the parameter provider gets called. We could add some logging etc. capability here,
                # but i am not sure
                return f_create_parameters(*args, **kwargs)
            return FunctionParameterProvider(name="custom_parameter_provider", fn=wrap_parameters)
        return parameter_decorator

    def dataset(self, *args, name=None, **kwargs):
        def dataset_decorator(f_create_dataset):
            @wraps(f_create_dataset)
            def wrap_dataset(*args, **kwargs):
                # here the workflow gets called. We could add some logging etc. capability here, but i am not sure
                return f_create_dataset(*args, **kwargs)

            if name is None:
                return self.datasets.load(wrap_dataset())
            return self.datasets.load(wrap_dataset(), name=name, **kwargs)

        return dataset_decorator

    def print(self, obj):
        if self.has_print():
            self.print_(obj)

    def print_tables(self, objects: List[Tablefyable], **kwargs):
        if self.has_print():
            self.print_("Loading.....")
            self.print_(to_table(objects, **kwargs))

    def has_print(self) -> bool:
        return self._print is None

    def print_(self, output, **kwargs):
        if self.has_print():
            self._print(output, **kwargs)


def example_app():
    config_path = os.path.join(os.path.expanduser("~"), ".padre-example.cfg")
    workspace_path = os.path.join(os.path.expanduser("~"), ".pypadre-example")

    """Create config file for testing purpose"""
    config = configparser.ConfigParser()
    with open(config_path, 'w+') as configfile:
        config.write(configfile)

    config = PadreConfig(config_file=config_path)
    config.set("backends", str([
        {
            "root_dir": workspace_path
        }
    ]))
    return PadreAppFactory.get(config)
