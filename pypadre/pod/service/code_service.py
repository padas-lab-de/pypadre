from typing import List

from pypadre.core.events.events import connect_subclasses
from pypadre.core.model.code.code import Code
from pypadre.pod.repository.i_repository import ICodeRepository
from pypadre.pod.service.base_service import BaseService


class CodeService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[ICodeRepository], **kwargs):
        super().__init__(model_clz=Code, backends=backends, **kwargs)

        @connect_subclasses(Code)
        def put(obj, **kwargs):
            self.put(obj, **kwargs)
        self.save_signal_fn(put)

        @connect_subclasses(Code)
        def delete(obj, **kwargs):
            self.delete(obj)
        self.save_signal_fn(delete)
