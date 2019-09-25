from typing import List

from pypadre.pod.repository.i_repository import ISplitRepository
from pypadre.pod.service.base_service import BaseService


class SplitService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[ISplitRepository], **kwargs):
        super().__init__(backends=backends, **kwargs)
