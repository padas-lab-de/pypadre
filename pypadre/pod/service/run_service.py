from typing import List

from pypadre.core.events.events import connect
from pypadre.core.model.computation.run import Run
from pypadre.pod.repository.i_repository import IRunRepository
from pypadre.pod.service.base_service import BaseService


class RunService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[IRunRepository], **kwargs):
        super().__init__(model_clz=Run, backends=backends, **kwargs)

    @connect(Run)
    def put(self, obj):
        super().put(obj)

    @connect(Run)
    def delete(self, obj):
        super().delete(obj)
