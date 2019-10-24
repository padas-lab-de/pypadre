from typing import List

from pypadre.core.metrics.metrics import IMetricProvider, Metric
from pypadre.core.model.computation.computation import Computation
from pypadre.core.model.generic.i_model_mixins import ILoggable


class MeasureService(ILoggable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    providers = {}
    providers_by_consumption = {}

    def add_measure(self, provider: IMetricProvider):
        if provider.name in self.providers.keys():
            self.send_warn("Measure already defined. Omitted adding it to the measure service: " + str(provider))
        else:
            self.providers[provider.name] = provider
            if not hasattr(self.providers_by_consumption, provider.consumes):
                self.providers_by_consumption[provider.consumes] = [provider]
            else:
                self.providers_by_consumption[provider.consumes].append(provider)

    def available_measures(self, computation: Computation) -> List[IMetricProvider]:
        measures = set()
        available_formats = set(computation.format)

        for data_format in available_formats:
            if hasattr(self.providers_by_consumption, data_format):
                for provider in self.providers_by_consumption[data_format]:
                    measures.add(provider)


        pass

    def calculate_measures(self, computation: Computation, **kwargs) -> List[Metric]:
        pass


measure_service = MeasureService()
