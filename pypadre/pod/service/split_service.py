from typing import List

from pypadre.core.events.events import connect
from pypadre.core.model.split.split import Split
from pypadre.pod.repository.i_repository import ISplitRepository
from pypadre.pod.service.base_service import BaseService


class SplitService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[ISplitRepository], **kwargs):
        super().__init__(model_clz=Split, backends=backends, **kwargs)

    @connect(Split)
    def put(self, obj):
        super().put(obj)

    @connect(Split)
    def delete(self, obj):
        super().delete(obj)
