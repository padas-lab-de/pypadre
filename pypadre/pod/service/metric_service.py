from typing import List

from pypadre.core.events.events import connect_subclasses, connect, CommonSignals
from pypadre.core.metrics.metrics import MetricProviderMixin, Metric
from pypadre.core.model.split.split import Split
from pypadre.pod.repository.i_repository import IMetricRepository
from pypadre.pod.service.base_service import ModelServiceMixin


class MetricService(ModelServiceMixin):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, measure_meters: List[MetricProviderMixin], backends: List[IMetricRepository], **kwargs):
        super().__init__(model_clz=Split, backends=backends, **kwargs)
        self._measure_meters = measure_meters

        @connect(Metric)
        @connect_subclasses(Metric, name=CommonSignals.PUT.name)
        def put(obj, **kwargs):
            self.put(obj)
        self.save_signal_fn(put)

    @property
    def measure_meters(self):
        return self._measure_meters

    def get_for_run(self, run):
        return self.list({Metric.RUN_ID: run.id})

    def gather_for_run(self, run):
        return self.list({Metric.RUN_ID: run.id})

    def gather_for_computation(self, computation):
        return computation

    # TODO look up which measure meters are applicable (This should be done by owl ontology, format, etc)
    # TODO we want also to allow for measure meter chains here. For example computing confusion matrix and
    # following up with roc curve points etc.
