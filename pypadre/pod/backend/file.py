import os

from pypadre.core.events.events import base_signals
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.local.file.dataset_repository import DatasetFileRepository
from pypadre.pod.repository.local.file.execution_repository import ExecutionFileRepository
from pypadre.pod.repository.local.file.experiment_repository import ExperimentFileRepository
from pypadre.pod.repository.local.file.project_repository import ProjectFileRepository
from pypadre.pod.repository.local.file.run_repository import RunFileRepository
from pypadre.pod.repository.local.file.split_repository import SplitFileRepository


class PadreFileBackend(IPadreBackend):
    """
    Backend class holding the repositories for our padre objects.

    root_dir
      |------datasets\
      |------experiments\
    """

    def log_info(self, message, **kwargs):
        self.log(message="INFO: " + message, **kwargs)

    def log_warn(self, message, **kwargs):
        self.log(message="WARN: " + message, **kwargs)

    def log_error(self, message, **kwargs):
        self.log(message="ERROR: " + message, **kwargs)

    def log(self, message, **kwargs):
        if self._file is None:
            self._file = open(os.path.join(self.root_dir, "padre.log"), "a")
        self._file.write(message)

    def __init__(self, config):
        super().__init__(config)
        self._dataset = DatasetFileRepository(self)
        self._project = ProjectFileRepository(self)
        self._experiment = ExperimentFileRepository(self)
        self._execution = ExecutionFileRepository(self)
        self._run = RunFileRepository(self)
        self._split = SplitFileRepository(self)

        # logging
        self._file = None

    @property
    def dataset(self) -> DatasetFileRepository:
        return self._dataset

    @property
    def project(self) -> ProjectFileRepository:
        return self._project

    @property
    def experiment(self) -> ExperimentFileRepository:
        return self._experiment

    @property
    def execution(self) -> ExecutionFileRepository:
        return self._execution

    @property
    def run(self) -> RunFileRepository:
        return self._run

    @property
    def split(self) -> SplitFileRepository:
        return self._split
