from typing import List

from pypadre.app.base_app import BaseChildApp
from pypadre.pod.service.split_service import SplitService
from pypadre.pod.backend.interfaces.backend.i_split_backend import ISplitBackend


class SplitApp(BaseChildApp):

    def __init__(self, parent, backends: List[ISplitBackend], **kwargs):
        super().__init__(parent, service=SplitService(backends=backends), **kwargs)
