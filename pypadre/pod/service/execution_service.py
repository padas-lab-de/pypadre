from typing import List

from pypadre.core.events.events import connect
from pypadre.core.model.execution import Execution
from pypadre.pod.repository.i_repository import IExecutionRepository
from pypadre.pod.service.base_service import BaseService


class ExecutionService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[IExecutionRepository], **kwargs):
        super().__init__(model_clz=Execution, backends=backends, **kwargs)

        @connect(Execution)
        def put(obj, **kwargs):
            self.put(obj)
        self.save_signal_fn(put)

        @connect(Execution)
        def delete(obj, **kwargs):
            self.delete(obj)
        self.save_signal_fn(delete)
