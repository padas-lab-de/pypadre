from typing import List

from pypadre.core.validation.validation import ValidateableFactory
from pypadre.pod.app.base_app import BaseChildApp
from pypadre.pod.repository.i_repository import ICodeRepository
from pypadre.pod.service.code_service import CodeService


class CodeApp(BaseChildApp):

    def __init__(self, parent, backends: List[ICodeRepository], **kwargs):
        super().__init__(parent, service=CodeService(backends=backends), **kwargs)

    def create(self, *args, clz, handlers=None, **kwargs):
        if handlers is None:
            handlers = []
        return ValidateableFactory.make(clz, *args, handlers=handlers, **kwargs)

    def put(self, obj, **kwargs):
        return self.service.put(obj, **kwargs)
