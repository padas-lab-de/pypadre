from typing import List

from pypadre.core.events.events import connect
from pypadre.core.model.computation.computation import Computation
from pypadre.core.model.split.split import Split
from pypadre.pod.repository.i_repository import IComputationRepository
from pypadre.pod.service.base_service import ModelServiceMixin


class ComputationService(ModelServiceMixin):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[IComputationRepository], **kwargs):
        super().__init__(model_clz=Split, backends=backends, **kwargs)

        @connect(Computation)
        # @connect_subclasses(Computation)
        def put(obj, **kwargs):
            self.put(obj, **kwargs)
        self.save_signal_fn(put)

        @connect(Computation)
        # @connect_subclasses(Computation)
        def delete(obj, **kwargs):
            self.delete(obj)
        self.save_signal_fn(delete)
