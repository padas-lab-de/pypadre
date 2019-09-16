from typing import List

from pypadre.pod.service.base_service import BaseService
from pypadre.pod.backend.interfaces.backend.i_project_backend import IProjectBackend


class ProjectService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[IProjectBackend], **kwargs):
        super().__init__(backends=backends, **kwargs)
