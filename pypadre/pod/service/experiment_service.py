from typing import List

from pypadre.pod.repository.i_repository import IExperimentRepository
from pypadre.pod.service.base_service import BaseService


class ExperimentService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[IExperimentRepository], **kwargs):
        super().__init__(backends=backends, **kwargs)

    def execute(self, id):
        experiment = self.get(id)
        return experiment.execute()
