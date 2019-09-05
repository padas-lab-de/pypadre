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
import ast
from typing import List

from jsonschema import ValidationError

from pypadre.app.base_app import IBaseApp
from pypadre.app.config.padre_config import PadreConfig
from pypadre.app.dataset.dataset_app import DatasetApp
from pypadre.app.metric_app import MetricApp
from pypadre.app.project.experiment.execution.execution_app import ExecutionApp
from pypadre.app.project.experiment.execution.run.run_app import RunApp
from pypadre.app.project.experiment.execution.run.split.split_app import SplitApp
from pypadre.app.project.experiment.experiment_app import ExperimentApp
from pypadre.app.project.project_app import ProjectApp
from pypadre.backend.interfaces.backend.i_backend import IBackend
from pypadre.backend.interfaces.backend.i_padre_backend import IPadreBackend
from pypadre.backend.local.file.file import PadreFileBackend
from pypadre.backend.remote.http.http import PadreHttpBackend
from pypadre.base import PadreLogger
from pypadre.eventhandler import add_logger
from pypadre.printing.tablefyable import Tablefyable
from pypadre.printing.util.print_util import to_table

logger = PadreLogger(app=None)
add_logger(logger=logger)


class PadreFactory:

    @staticmethod
    def get(config=PadreConfig()):
        backends = PadreFactory._parse_backends(config)
        return PadreApp(backends=backends)

    @staticmethod
    def _parse_backends(config):
        _backends = config.get("backends", "GENERAL")

        backends = []
        for b in _backends:
            if 'base_url' in b:
                # TODO check for validity
                backends.append(PadreHttpBackend(b))
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
        super().__init__(backends=backends)
        self._print = printer

        if backends is None:
            pass
            # TODO inform the user

        # TODO Should each subApp really hold each backend? This may be convenient to code like this.
        self._dataset_app = DatasetApp(self, [backend.dataset for backend in backends] if backends is not None else None)
        self._project_app = ProjectApp(self, [backend.project for backend in backends] if backends is not None else None)
        self._experiment_app = ExperimentApp(self, [backend.project.experiment for backend in backends] if backends is not None else None)
        self._execution_app = ExecutionApp(self, [backend.project.experiment.execution for backend in backends] if backends is not None else None)
        self._run_app = RunApp(self, [backend.project.experiment.execution.run for backend in backends] if backends is not None else None)
        self._split_app = SplitApp(self, [backend.project.experiment.execution.run.split for backend in backends] if backends is not None else None)
        #self._metric_app = MetricApp(self, [backend.project.experiment.execution.split.metric for backend in backends] if backends is not None else None)

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

# # TODO DELETE AS SOON AS TRANSFERED INTO NEW APP CLASSES
# class OldPadreApp:
#
#     # todo improve printing. Configure a proper printer or find a good ascii printing package
#
#     def __init__(self, config=None, printer=None, backends=None):
#         # Set default config if none is given
#         if config is None:
#             self._config = PadreConfig()
#         else:
#             self._config = config
#
#         self._print = printer
#         self._backend_app = BackEndApp(self._config)
#         self._dataset_app = DatasetApp(self)
#         self._experiment_app = ExperimentApp(self)
#         self._project_app = ProjectApp(self)
#         self._experiment_creator = ExperimentCreator()
#         self._metrics_evaluator = CompareMetrics(root_path=self._config.local_backend_config.get('root_dir', None))
#         self._metrics_reevaluator = ReevaluationMetrics()
#
#     @property
#     def offline(self):
#         """
#         sets the current offline / online status of the app. Permanent changes need to be done via the config.
#         :return: True, if requests are not passed to the server
#         """
#         return "offline" not in self._config.get("offline", "GENERAL")
#
#     @offline.setter
#     def offline(self, offline):
#         self._config.set("offline", offline, "GENERAL")
#
#     @property
#     def datasets(self):
#         return self._dataset_app
#
#     @property
#     def experiments(self):
#         return self._experiment_app
#
#     @property
#     def experiment_creator(self):
#         return self._experiment_creator
#
#     @property
#     def metrics_evaluator(self):
#         return self._metrics_evaluator
#
#     @property
#     def metrics_reevaluator(self):
#         return self._metrics_reevaluator
#
#     @property
#     def config(self):
#         return self._config
#
#     def set_printer(self, printer):
#         """
#         sets the printer, i.e. the output of console text. If None, there will be not text output
#         :param printer: object with .print(str) function like sys.stdout or None
#         """
#         self._print = printer
#
#     def status(self):
#         """
#         returns the status of the app, i.e. if the server is running, the client, the config etc.
#         :return:
#         """
#         pass
#
#     def print(self, output, **kwargs):
#         if self.has_print():
#             self._print(output, **kwargs)
#
#     def has_print(self):
#         return self._print is not None
#
#     @property
#     def remote_backend(self):
#         return self._http_repo
#
#     @property
#     def local_backend(self):
#         return self._file_repo
#
#     @property
#     def repository(self):
#         return self._dual_repo
#
#     def authenticate(self, user, passwd):
#         """
#         Authenticate user to the server
#         If authenticated successfully then set app and http client online and save token in config
#
#         :param user: User name
#         :param passwd: password for given user
#         :return: Token
#         """
#         token = self.remote_backend.authenticate(passwd, user)
#         if token is not None:
#             self.config.set('token', token)
#             self.config.save()
#             self.offline = False
#             self.config.general["offline"] = self.offline
#             self.remote_backend.online = not self.offline
#         return token


p_app = PadreApp(printer=print)  # load the default app
