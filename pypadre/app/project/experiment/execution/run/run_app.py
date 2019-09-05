from typing import List

from pypadre.app.base_app import BaseChildApp
from pypadre.core.backend.interfaces.backend.i_backend import IBackend


class RunApp(BaseChildApp):

    def __init__(self, parent, backends: List[IBackend], **kwargs):
        super().__init__(parent, backends, **kwargs)