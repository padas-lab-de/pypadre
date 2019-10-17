from typing import List

from pypadre.core.events.events import connect
from pypadre.core.model.experiment import Experiment
from pypadre.pod.repository.i_repository import IExperimentRepository
from pypadre.pod.service.base_service import BaseService


class ExperimentService(BaseService):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[IExperimentRepository], **kwargs):
        super().__init__(model_clz=Experiment, backends=backends, **kwargs)

        @connect(Experiment)
        def put(obj, **kwargs):
            self.put(obj)
        self.save_signal_fn(put)

        @connect(Experiment)
        def delete(obj, **kwargs):
            self.delete(obj)
        self.save_signal_fn(delete)

    def execute(self, id):
        experiment = self.get(id)
        return experiment.execute()
