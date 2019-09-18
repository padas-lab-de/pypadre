from typing import List

from pypadre.pod.service.base_service import BaseService
from pypadre.pod.repository.i_repository import IRunRepository


class RunService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[IRunRepository], **kwargs):
        super().__init__(backends=backends, **kwargs)
