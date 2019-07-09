"""
Padre app as single point of interaction.

Defaults:

- The default configuration is provided under `.padre.cfg` in the user home directory


Architecture of the module
++++++++++++++++++++++++++

- `PadreConfig` wraps the configuration for the app. It can read/write the config from a file (if provided)

"""


# todo merge with cli. cli should use app and app should be configurable via builder pattern and configuration files
from pypadre.app.backend.backend_app import BackEndApp
from pypadre.app.config.padre_config import PadreConfig
from pypadre.app.dataset.dataset_app import DatasetApp
from pypadre.app.experiment.experiment_app import ExperimentApp
from pypadre.app.project.project_app import ProjectApp
from pypadre.base import PadreLogger
from pypadre.eventhandler import add_logger
from pypadre.experimentcreator import ExperimentCreator
from pypadre.metrics import CompareMetrics
from pypadre.metrics import ReevaluationMetrics

logger = PadreLogger()
add_logger(logger=logger)


class PadreApp:

    # todo improve printing. Configure a proper printer or find a good ascii printing package

    def __init__(self, config=None, printer=None, backends=None):
        # Set default config if none is given
        if config is None:
            self._config = PadreConfig()
        else:
            self._config = config

        self._print = printer
        self._backend_app = BackEndApp(self._config)
        self._dataset_app = DatasetApp(self)
        self._experiment_app = ExperimentApp(self)
        self._project_app = ProjectApp(self)
        self._experiment_creator = ExperimentCreator()
        self._metrics_evaluator = CompareMetrics(root_path=self._config.local_backend_config.get('root_dir', None))
        self._metrics_reevaluator = ReevaluationMetrics()

    @property
    def offline(self):
        """
        sets the current offline / online status of the app. Permanent changes need to be done via the config.
        :return: True, if requests are not passed to the server
        """
        return "offline" not in self._config.get("offline", "GENERAL")

    @offline.setter
    def offline(self, offline):
        self._config.set("offline", offline, "GENERAL")

    @property
    def datasets(self):
        return self._dataset_app

    @property
    def experiments(self):
        return self._experiment_app

    @property
    def experiment_creator(self):
        return self._experiment_creator

    @property
    def metrics_evaluator(self):
        return self._metrics_evaluator

    @property
    def metrics_reevaluator(self):
        return self._metrics_reevaluator

    @property
    def config(self):
        return self._config

    def set_printer(self, printer):
        """
        sets the printer, i.e. the output of console text. If None, there will be not text output
        :param printer: object with .print(str) function like sys.stdout or None
        """
        self._print = printer

    def status(self):
        """
        returns the status of the app, i.e. if the server is running, the client, the config etc.
        :return:
        """
        pass

    def print(self, output, **kwargs):
        if self.has_print():
            self._print(output, **kwargs)

    def has_print(self):
        return self._print is not None

    @property
    def remote_backend(self):
        return self._http_repo

    @property
    def local_backend(self):
        return self._file_repo

    @property
    def repository(self):
        return self._dual_repo

    def authenticate(self, user, passwd):
        """
        Authenticate user to the server
        If authenticated successfully then set app and http client online and save token in config

        :param user: User name
        :param passwd: password for given user
        :return: Token
        """
        token = self.remote_backend.authenticate(passwd, user)
        if token is not None:
            self.config.set('token', token)
            self.config.save()
            self.offline = False
            self.config.general["offline"] = self.offline
            self.remote_backend.online = not self.offline
        return token


p_app = PadreApp(printer=print)  # load the default app
