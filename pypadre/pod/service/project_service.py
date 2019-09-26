from typing import List

from pypadre.core.events.events import connect
from pypadre.core.model.project import Project
from pypadre.pod.repository.i_repository import IProjectRepository
from pypadre.pod.service.base_service import BaseService


class ProjectService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[IProjectRepository], **kwargs):
        super().__init__(model_clz=Project, backends=backends, **kwargs)

    @connect(Project)
    def put(self, obj):
        super().put(obj)

    @connect(Project)
    def delete(self, obj):
        super().delete(obj)
