from typing import List

from pypadre.pod.repository.i_repository import IExecutionRepository
from pypadre.pod.service.base_service import BaseService


class ExecutionService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[IExecutionRepository], **kwargs):
        super().__init__(backends=backends, **kwargs)
