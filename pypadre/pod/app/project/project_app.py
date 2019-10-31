from typing import List

from pypadre.pod.app.base_app import BaseChildApp
from pypadre.pod.repository.i_repository import IProjectRepository
from pypadre.pod.service.project_service import ProjectService


class ProjectApp(BaseChildApp):
    def __init__(self, parent, backends: List[IProjectRepository], **kwargs):
        super().__init__(parent, service=ProjectService(backends=backends), **kwargs)

    def create(self, *args, **kwargs):
        """
        Function creates a project and puts the project to the backend
        :return:
        """
        project = self.service.create(*args, **kwargs)
        self.put(project)
        return project
