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

    def execute(self, id):
        experiment = self.get(id)
        return experiment.execute()

    @connect(Experiment)
    def put(self, obj):
        super().put(obj)

    @connect(Experiment)
    def delete(self, obj):
        super().delete(obj)
