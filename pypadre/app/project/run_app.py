from typing import List

from pypadre.app.base_app import BaseChildApp
from pypadre.pod.repository.i_repository import IRunRepository
from pypadre.pod.service.run_service import RunService


class RunApp(BaseChildApp):

    def __init__(self, parent, backends: List[IRunRepository], **kwargs):
        super().__init__(parent, service=RunService(backends=backends), **kwargs)
