from typing import List

from pypadre.pod.app.base_app import BaseChildApp
from pypadre.pod.repository.i_repository import IPipelineOutputRepository
from pypadre.pod.service.pipeline_output_service import PipelineOutputService


class PipelineOutputApp(BaseChildApp):

    def __init__(self, parent, backends: List[IPipelineOutputRepository], **kwargs):
        super().__init__(parent, service=PipelineOutputService(backends=backends), **kwargs)
