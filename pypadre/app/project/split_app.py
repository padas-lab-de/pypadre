from typing import List

from pypadre.app.base_app import BaseChildApp
from pypadre.pod.service.split_service import SplitService
from pypadre.pod.repository.i_repository import ISplitRepository


class SplitApp(BaseChildApp):

    def __init__(self, parent, backends: List[ISplitRepository], **kwargs):
        super().__init__(parent, service=SplitService(backends=backends), **kwargs)
