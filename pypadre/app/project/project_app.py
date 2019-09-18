from typing import List

from pypadre.app.base_app import BaseChildApp
from pypadre.pod.service.project_service import ProjectService
from pypadre.pod.repository.i_repository import IProjectRepository


class ProjectApp(BaseChildApp):
    def __init__(self, parent, backends: List[IProjectRepository], **kwargs):
        super().__init__(parent, service=ProjectService(backends=backends), **kwargs)
