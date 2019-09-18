from typing import List

from pypadre.app.base_app import BaseChildApp
from pypadre.pod.service.execution_service import ExecutionService
from pypadre.pod.repository.i_repository import IExecutionRepository


class ExecutionApp(BaseChildApp):

    def __init__(self, parent, backends: List[IExecutionRepository], **kwargs):
        super().__init__(parent, service=ExecutionService(backends=backends), **kwargs)
