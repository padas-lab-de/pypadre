from typing import List

from pypadre.pod.app.base_app import BaseChildApp
from pypadre.pod.repository.i_repository import ICodeRepository
from pypadre.pod.service.code_service import CodeService


class CodeApp(BaseChildApp):

    def __init__(self, parent, backends: List[ICodeRepository], **kwargs):
        super().__init__(parent, service=CodeService(backends=backends), **kwargs)
