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
from typing import List

from jsonschema import ValidationError

from pypadre.core.printing.tablefyable import Tablefyable
from pypadre.core.printing.util.print_util import to_table
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
from pypadre.pod.service.logging_service import LoggingService

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
        self._logger = LoggingService(backends)
        self._dataset_app = DatasetApp(self, [backend.dataset for backend in backends] if backends is not None else None)
        self._project_app = ProjectApp(self, [backend.project for backend in backends] if backends is not None else None)
        self._experiment_app = ExperimentApp(self, [backend.experiment for backend in backends] if backends is not None else None)
        self._execution_app = ExecutionApp(self, [backend.execution for backend in backends] if backends is not None else None)
        self._run_app = RunApp(self, [backend.run for backend in backends] if backends is not None else None)
        self._split_app = SplitApp(self, [backend.split for backend in backends] if backends is not None else None)
        self._computation_app = ComputationApp(self, [backend.computation for backend in backends] if backends is not None else None)
        self._metric_app = MetricApp(self, [backend.metric for backend in backends] if backends is not None else None)
        self._code_app = CodeApp(self, [backend.code for backend in backends] if backends is not None else None)
        self._pipeline_output_app = PipelineOutputApp(self, [backend.pipeline_output for backend in backends] if backends is not None else None)


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
