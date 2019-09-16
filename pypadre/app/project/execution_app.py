from typing import List

from pypadre.app.base_app import BaseChildApp
from pypadre.pod.service.execution_service import ExecutionService
from pypadre.pod.backend.interfaces.backend.i_execution_backend import IExecutionBackend


class ExecutionApp(BaseChildApp):

    def __init__(self, parent, backends: List[IExecutionBackend], **kwargs):
        super().__init__(parent, service=ExecutionService(backends=backends), **kwargs)
