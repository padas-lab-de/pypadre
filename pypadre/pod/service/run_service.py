from typing import List

from pypadre.pod.service.base_service import BaseService
from pypadre.pod.backend.interfaces.backend.i_run_backend import IRunBackend


class RunService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[IRunBackend], **kwargs):
        super().__init__(backends=backends, **kwargs)
