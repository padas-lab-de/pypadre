from typing import List

from pypadre.app.base_app import BaseChildApp
from pypadre.core.service.project_service import ProjectService
from pypadre.pod.backend.interfaces.backend.i_backend import IBackend
from pypadre.pod.backend.interfaces.backend.i_project_backend import IProjectBackend


class ProjectApp(BaseChildApp):
    def __init__(self, parent, backends: List[IProjectBackend], **kwargs):
        super().__init__(parent, service=ProjectService(backends=backends), **kwargs)
