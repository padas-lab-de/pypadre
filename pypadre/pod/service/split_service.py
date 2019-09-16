from typing import List

from pypadre.pod.service.base_service import BaseService
from pypadre.pod.backend.interfaces.backend.i_split_backend import ISplitBackend


class SplitService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[ISplitBackend], **kwargs):
        super().__init__(backends=backends, **kwargs)
