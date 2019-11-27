from typing import List

from pypadre.core.printing.util.print_util import to_table
from pypadre.pod.app.base_app import IBaseApp
from pypadre.pod.app.code_app import CodeApp
from pypadre.pod.app.dataset.dataset_app import DatasetApp
from pypadre.pod.app.metric_app import MetricApp
from pypadre.pod.app.project.computation_app import ComputationApp
from pypadre.pod.app.project.execution_app import ExecutionApp
from pypadre.pod.app.project.experiment_app import ExperimentApp
from pypadre.pod.app.project.pipeline_output_app import PipelineOutputApp
from pypadre.pod.app.project.project_app import ProjectApp
from pypadre.pod.app.project.run_app import RunApp
from pypadre.pod.app.project.split_app import SplitApp
from pypadre.pod.backend.i_padre_backend import IPadreBackend


class CoreApp(IBaseApp):

    # TODO metric algorithms should be passed for metric calculation. This should work a bit like on the server. Metrics themselves are plugins which are invoked by the reevaluater
    def __init__(self, printer=None, backends: List[IPadreBackend] = None):
        super().__init__()

        if printer is None:
            self._print = print

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

    def has_print(self):
        return self._print is not None

    def print(self, obj):
        self.print_(obj)

    def print_tables(self, clz, objects, **kwargs):
        self.print_("Loading " + clz.__name__ + " table...")
        self.print_(to_table(clz, objects, **kwargs))

    def print_(self, output, **kwargs):
        if self.has_print():
            self._print(output, **kwargs)
