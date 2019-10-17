from typing import List

from pypadre.pod.app.base_app import BaseChildApp
from pypadre.pod.repository.i_repository import IMetricRepository
from pypadre.pod.service.metric_service import MetricService


class MetricApp(BaseChildApp):

    def __init__(self, parent, backends: List[IMetricRepository], **kwargs):
        super().__init__(parent, service=MetricService(measure_meters=[], backends=backends), **kwargs)
