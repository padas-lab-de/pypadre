from typing import List

from pypadre.app.base_app import BaseChildApp
from pypadre.pod.service.run_service import RunService
from pypadre.pod.backend.interfaces.backend.i_run_backend import IRunBackend


class RunApp(BaseChildApp):

    def __init__(self, parent, backends: List[IRunBackend], **kwargs):
        super().__init__(parent, service=RunService(backends=backends), **kwargs)
