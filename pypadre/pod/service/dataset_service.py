from typing import List

from pypadre.pod.repository.i_repository import IDatasetRepository
from pypadre.pod.service.base_service import BaseService


class DatasetService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[IDatasetRepository], **kwargs):
        super().__init__(backends=backends, **kwargs)
