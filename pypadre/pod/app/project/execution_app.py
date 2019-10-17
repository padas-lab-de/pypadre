from typing import List

from pypadre.pod.app.base_app import BaseChildApp
from pypadre.pod.repository.i_repository import IExecutionRepository
from pypadre.pod.service.execution_service import ExecutionService


class ExecutionApp(BaseChildApp):

    def __init__(self, parent, backends: List[IExecutionRepository], **kwargs):
        super().__init__(parent, service=ExecutionService(backends=backends), **kwargs)
