from typing import List

from pypadre.app.base_app import BaseChildApp
from pypadre.pod.service.run_service import RunService
from pypadre.pod.repository.i_repository import IRunRepository


class RunApp(BaseChildApp):

    def __init__(self, parent, backends: List[IRunRepository], **kwargs):
        super().__init__(parent, service=RunService(backends=backends), **kwargs)
