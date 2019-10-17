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
        def put(obj, **kwargs):
            self.put(obj)
        self.save_signal_fn(put)

        @connect(Split)
        def delete(obj, **kwargs):
            self.delete(obj)
        self.save_signal_fn(delete)
