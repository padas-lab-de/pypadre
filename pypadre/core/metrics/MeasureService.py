from typing import List

from pypadre.core.metrics.metrics import MeasureMeter, Metric
from pypadre.core.model.computation.computation import Computation
from pypadre.core.model.generic.i_model_mixins import ILoggable


class MeasureService(ILoggable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    measures = {}

    def add_measure(self, measure_meter: MeasureMeter):
        if measure_meter.name in self.measures.keys():
            self.send_warn("Measure already defined. Omitted adding it to the measure service: " + str(measure_meter))
        else:
            self.measures[measure_meter.name] = measure_meter

    def available_measures(self, computation: Computation) -> List[MeasureMeter]:
        # TODO work with trees
        pass

    def calculate_measures(self, computation: Computation, **kwargs) -> List[Metric]:
        pass


measure_service = MeasureService()
