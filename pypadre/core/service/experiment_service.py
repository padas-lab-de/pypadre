from typing import List

from pypadre.core.service.base_service import BaseService
from pypadre.pod.backend.interfaces.backend.i_dataset_backend import IDatasetBackend
from pypadre.pod.backend.interfaces.backend.i_experiment_backend import IExperimentBackend


class ExperimentService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[IExperimentBackend], **kwargs):
        super().__init__(backends=backends, **kwargs)
