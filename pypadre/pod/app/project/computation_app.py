from typing import List

from pypadre.pod.app.base_app import BaseChildApp
from pypadre.pod.repository.i_repository import IComputationRepository
from pypadre.pod.service.computation_service import ComputationService


class ComputationApp(BaseChildApp):

    def __init__(self, parent, backends: List[IComputationRepository], **kwargs):
        super().__init__(parent, service=ComputationService(backends=backends), **kwargs)
