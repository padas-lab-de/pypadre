from typing import List

from pypadre.core.events.events import connect_subclasses
from pypadre.core.model.code.icode import ICode
from pypadre.core.model.generic.i_model_mixins import IStoreable
from pypadre.pod.repository.i_repository import ICodeRepository
from pypadre.pod.service.base_service import BaseService


class CodeService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[ICodeRepository], **kwargs):
        super().__init__(model_clz=ICode, backends=backends, **kwargs)

        @connect_subclasses(ICode)
        def put(obj, **sended_kwargs):
            self.put(obj, **sended_kwargs)
        self.save_signal_fn(put)

        @connect_subclasses(ICode)
        def delete(obj, **sended_kwargs):
            self.delete(obj)
        self.save_signal_fn(delete)

        @connect_subclasses(ICode)
        def get(sender, **sended_kwargs):
            return_val = sended_kwargs.get(IStoreable.RETURN_VAL)
            name = sended_kwargs.get("name")
            setattr(return_val, IStoreable.RETURN_VAL, self.get(name))
        self.save_signal_fn(get)
