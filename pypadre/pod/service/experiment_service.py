from typing import List

from pypadre.pod.service.base_service import BaseService
from pypadre.pod.repository.i_repository import IExperimentRepository


class ExperimentService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[IExperimentRepository], **kwargs):
        super().__init__(backends=backends, **kwargs)
