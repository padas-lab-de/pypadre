from typing import List

from pypadre.core.events.events import connect
from pypadre.core.metrics.metrics import IMetricProvider
from pypadre.core.model.computation.computation import Computation
from pypadre.core.model.split.split import Split
from pypadre.pod.repository.i_repository import IMetricRepository
from pypadre.pod.service.base_service import BaseService


class MetricService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, measure_meters: List[IMetricProvider], backends: List[IMetricRepository], **kwargs):
        super().__init__(model_clz=Split, backends=backends, **kwargs)
        self._measure_meters = measure_meters

        @connect(Computation, name="put")
        def metrics(obj, **kwargs):
            print("Calcuate metrics! " + str(obj))
        self.save_signal_fn(metrics)

    @property
    def measure_meters(self):
        return self._measure_meters

    # TODO look up which measure meters are applicable (This should be done by owl ontology, format, etc)
    # TODO we want also to allow for measure meter chains here. For example computing confusion matrix and
    # following up with roc curve points etc.
