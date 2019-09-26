from typing import List

from pypadre.core.events.events import connect
from pypadre.core.model.dataset.dataset import Dataset
from pypadre.pod.repository.i_repository import IDatasetRepository
from pypadre.pod.service.base_service import BaseService


class DatasetService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[IDatasetRepository], **kwargs):
        super().__init__(model_clz=Dataset, backends=backends, **kwargs)

    @connect(Dataset)
    def put(self, obj):
        super().put(obj)

    @connect(Dataset)
    def delete(self, obj):
        super().delete(obj)
