import os

from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import IProjectRepository
from pypadre.pod.repository.remote.gitlab.project_repository import ProjectGitlabRepository


class PadreGitLabBackend(IPadreBackend):
    """
    backend class holding the gitlab repositories for our padre objects
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
        #TODO finsh all backends repos
        self._project = ProjectGitlabRepository(self)
        # logging
        self._file = None

    @property
    def project(self) -> IProjectRepository:
        return self._project

