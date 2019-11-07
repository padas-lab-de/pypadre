from typing import List

from pypadre.core.events.events import connect_subclasses, connect
from pypadre.core.model.code.code_file import CodeFile
from pypadre.core.model.code.codemixin import CodeMixin
from pypadre.core.model.generic.i_model_mixins import StoreableMixin
from pypadre.pod.repository.i_repository import ICodeRepository
from pypadre.pod.service.base_service import ModelServiceMixin


class CodeService(ModelServiceMixin):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[ICodeRepository], **kwargs):
        super().__init__(model_clz=CodeMixin, backends=backends, **kwargs)

        @connect_subclasses(CodeMixin)
        def put(obj, **sended_kwargs):
            self.put(obj, **sended_kwargs)
        self.save_signal_fn(put)

        @connect_subclasses(CodeMixin)
        def delete(obj, **sended_kwargs):
            self.delete(obj)
        self.save_signal_fn(delete)

        @connect_subclasses(CodeMixin)
        def get(sender, **sended_kwargs):
            return_val = sended_kwargs.get(StoreableMixin.RETURN_VAL)
            name = sended_kwargs.get("name")
            setattr(return_val, StoreableMixin.RETURN_VAL, self.get(name))
        self.save_signal_fn(get)

        @connect(CodeFile)
        def codehash(obj, **sended_kwargs):
            for b in backends:
                if hasattr(b, 'get_code_hash'):
                    b.get_code_hash(obj=obj, **sended_kwargs)

        self.save_signal_fn(codehash)

