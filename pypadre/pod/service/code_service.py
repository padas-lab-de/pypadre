from typing import List

from pypadre.core.events.events import connect_subclasses, CommonSignals
from pypadre.core.model.code.code_mixin import CodeMixin
from pypadre.pod.repository.i_repository import ICodeRepository
from pypadre.pod.service.base_service import ModelServiceMixin


class CodeService(ModelServiceMixin):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[ICodeRepository], **kwargs):
        super().__init__(model_clz=CodeMixin, backends=backends, **kwargs)

        @connect_subclasses(CodeMixin, name=CommonSignals.PUT.name)
        def put(obj, **sended_kwargs):
            self.put(obj, **sended_kwargs)
        self.save_signal_fn(put)

        @connect_subclasses(CodeMixin, name=CommonSignals.DELETE.name)
        def delete(obj, **sended_kwargs):
            self.delete(obj)
        self.save_signal_fn(delete)

        # @connect_subclasses(CodeMixin, name=CommonSignals.GET.name)
        # def get_by_name(sender, **sended_kwargs):
        #     """
        #     Function to get an code object by name.
        #     :param sender:
        #     :param sended_kwargs:
        #     :return:
        #     """
        #     return_val = sended_kwargs.get(StoreableMixin.RETURN_VAL)
        #     name = sended_kwargs.get("name")
        #     if name is not None:
        #         return_val[StoreableMixin.RETURN_VAL] = next(iter(self.get(name)), None)
        # self.save_signal_fn(get_by_name)

        # @connect(CodeFile)
        # def codehash(obj, **sended_kwargs):
        #     for b in backends:
        #         if hasattr(b, 'get_code_hash'):
        #             b.get_code_hash(obj=obj, **sended_kwargs)
        #
        # self.save_signal_fn(codehash)

