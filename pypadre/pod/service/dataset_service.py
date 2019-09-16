from typing import List

from pypadre.pod.service.base_service import BaseService
from pypadre.pod.backend.interfaces.backend.i_dataset_backend import IDatasetBackend


class DatasetService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[IDatasetBackend], **kwargs):
        super().__init__(backends=backends, **kwargs)
