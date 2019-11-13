from typing import List

from pypadre.core.events.events import connect_subclasses, CommonSignals
from pypadre.core.model.computation.computation import Computation
from pypadre.pod.repository.i_repository import IComputationRepository
from pypadre.pod.service.base_service import ModelServiceMixin


class ComputationService(ModelServiceMixin):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[IComputationRepository], **kwargs):
        super().__init__(model_clz=Computation, backends=backends, **kwargs)

        @connect_subclasses(Computation, name=CommonSignals.PUT.name)
        # @connect_subclasses(Computation)
        def put(obj, **kwargs):
            self.put(obj, **kwargs)
        self.save_signal_fn(put)

        @connect_subclasses(Computation, name=CommonSignals.DELETE.name)
        # @connect_subclasses(Computation)
        def delete(obj, **kwargs):
            self.delete(obj)
        self.save_signal_fn(delete)
